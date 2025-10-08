import argparse
import time as t
from pathlib import Path

import cv2
import numpy as np

try:
    from current import process, sudoku
except Exception:
    import process, sudoku  # fallback if there's no 'current' package

try:
    from preprocessing import preprocess
except Exception:
    import preprocess  # fallback if there's no 'preprocessing' package

from tensorflow.keras.models import load_model


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def put_banner(img, text, y=30, color=(0, 255, 0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def show_if(flag, title, img):
    if flag and img is not None:
        # Auto-resize for visibility
        h, w = img.shape[:2]
        scale = 960.0 / max(w, 1)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imshow(title, img)


def to_uint8(img):
    # Normalize to [0,255] uint8 for inspection
    if img is None:
        return None
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
    return x


def squares_to_batch(squares):
    # Convert list/array of 2D or 3D squares to (N,32,32,1) float32 (0..255)
    arr = np.array(squares)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, -1)  # (N,32,32,1)
    if arr.ndim != 4:
        raise ValueError(f"[DIGITS] Expected (N,32,32,1) or (N,32,32)->expand, got shape={arr.shape}")
    return arr.astype(np.float32)  # let the model's Rescaling layer handle /255


def main():
    parser = argparse.ArgumentParser(description="Debug Sudoku solver pipeline with rich logging & views.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--model", type=str, default="digit_cnn.keras")
    parser.add_argument("--save-dir", type=str, default="debug_snaps")
    parser.add_argument("--invert-preview", action="store_true", help="Invert preview masks for better visibility")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    # Load model
    print(f"[MODEL] Loading: {args.model}")
    model = load_model(args.model)
    print("[MODEL] input_shape:", getattr(model, "input_shape", None))

    frameWidth, frameHeight = args.width, args.height
    cap = cv2.VideoCapture(args.camera)

    # Set camera props (some drivers ignore these; log the result)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

    ok_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ok_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ok_b = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print(f"[CAM] Opened={cap.isOpened()} size=({ok_w:.0f}x{ok_h:.0f}) brightness={ok_b}")

    # Timers and toggles
    prev = 0.0
    seen = dict()
    toggles = {
        "processed": True,
        "corners": True,
        "warped": True,
        "warped_processed": True,
        "mask": True,
        "numbers": True,
        "squares_grid": True,
        "result": True,
        "verbose": True,
    }

    help_text = (
        "Keys:\n"
        "  h  : toggle help overlay\n"
        "  1  : toggle processed frame\n"
        "  2  : toggle warped\n"
        "  3  : toggle mask\n"
        "  4  : toggle numbers\n"
        "  5  : toggle warped_processed\n"
        "  g  : toggle squares grid overlay\n"
        "  v  : toggle verbose logging\n"
        "  s  : save all current intermediate images to disk\n"
        "  q  : quit\n"
    )

    show_help = True
    snap_idx = 0

    while True:
        time_elapsed = t.time() - prev
        success, img = cap.read()
        if not success:
            print("[CAM] cap.read() failed. Exiting.")
            break

        if time_elapsed < 1.0 / max(args.fps, 1e-3):
            cv2.waitKey(1)
            continue

        prev = t.time()

        img_result = img.copy()
        img_corners = img.copy()

        try:
            processed_img = preprocess.preprocess(img)
            proc_u8 = to_uint8(processed_img)
            if toggles["verbose"]:
                print(f"[STEP] preprocess: shape={None if proc_u8 is None else proc_u8.shape}")
        except Exception as e:
            print("[ERR] preprocess failed:", e)
            processed_img = None
            proc_u8 = None

        corners = None
        try:
            corners = process.find_contours(processed_img, img_corners)
            if toggles["verbose"]:
                print(f"[STEP] find_contours: corners={corners}")
        except Exception as e:
            print("[ERR] find_contours failed:", e)

        warped = warped_processed = mask = numbers = None
        squares = squares_processed = None

        if corners:
            try:
                warped, matrix = process.warp_image(corners, img)
                warped_u8 = to_uint8(warped)
                if toggles["verbose"]:
                    print(f"[STEP] warp_image: warped.shape={warped_u8.shape if warped_u8 is not None else None}")
            except Exception as e:
                print("[ERR] warp_image failed:", e)
                warped_u8 = None

            try:
                warped_processed = preprocess.preprocess(warped)
                wp_u8 = to_uint8(warped_processed)
                if toggles["verbose"]:
                    print(f"[STEP] preprocess(warped): shape={wp_u8.shape if wp_u8 is not None else None}")
            except Exception as e:
                print("[ERR] preprocess(warped) failed:", e)
                wp_u8 = None

            try:
                vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
                mask = process.create_grid_mask(vertical_lines, horizontal_lines)
                msk_u8 = to_uint8(mask)
                if args.invert_preview and msk_u8 is not None:
                    msk_u8 = 255 - msk_u8
                if toggles["verbose"]:
                    print(f"[STEP] grid/mask: mask.shape={None if msk_u8 is None else msk_u8.shape}")
            except Exception as e:
                print("[ERR] grid/mask failed:", e)
                msk_u8 = None

            try:
                numbers = cv2.bitwise_and(warped_processed, mask) if (warped_processed is not None and mask is not None) else None
                nums_u8 = to_uint8(numbers)
                if toggles["verbose"]:
                    print(f"[STEP] numbers bitwise_and: shape={None if nums_u8 is None else nums_u8.shape}")
            except Exception as e:
                print("[ERR] numbers bitwise_and failed:", e)
                nums_u8 = None

            try:
                squares = process.split_into_squares(numbers)
                if toggles["verbose"]:
                    print(f"[STEP] split_into_squares: len={0 if squares is None else len(squares)} (expect 81)")
            except Exception as e:
                print("[ERR] split_into_squares failed:", e)

            try:
                squares_processed = process.clean_squares(squares)
                if toggles["verbose"] and squares_processed is not None:
                    print(f"[STEP] clean_squares: len={len(squares_processed)} sample_shape={np.array(squares_processed[0]).shape if len(squares_processed)>0 else None}")
            except Exception as e:
                print("[ERR] clean_squares failed:", e)

            # Recognition with safety
            squares_guesses = None
            if squares_processed:
                try:
                    # Try your existing recognizer first
                    squares_guesses = process.recognize_digits(squares_processed, model)
                    if toggles["verbose"]:
                        print(f"[STEP] recognize_digits(process): got string len={len(squares_guesses)} -> {squares_guesses[:20]}...")
                except Exception as e:
                    print("[ERR] recognize_digits failed, trying manual inference:", e)
                    try:
                        batch = squares_to_batch(squares_processed)  # (N,32,32,1) float32
                        probs = model.predict(batch, verbose=0)
                        preds = probs.argmax(axis=1) + 1  # 1..9
                        # Define 'empty' as near-black squares (mean < threshold). Adjust if needed:
                        empties = [int(np.mean(np.array(s)) < 10) for s in squares_processed]
                        digits = [0 if empties[i] else int(preds[i]) for i in range(len(preds))]
                        squares_guesses = "".join(str(d) for d in digits)
                        print("[STEP] manual inference OK. guesses length:", len(squares_guesses))
                    except Exception as ee:
                        print("[ERR] manual inference also failed:", ee)

                # Caching and solve if we have guesses
                if squares_guesses and len(squares_guesses) == 81:
                    if squares_guesses in seen and seen[squares_guesses] is False:
                        pass  # impossible; skip
                    elif squares_guesses in seen:
                        solved_puzzle, solve_time = seen[squares_guesses]
                        try:
                            process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                            img_result = process.unwarp_image(warped, img_result, corners, solve_time)
                        except Exception as e:
                            print("[ERR] draw/unwarp (cached) failed:", e)
                    else:
                        try:
                            solved_puzzle, solve_time = sudoku.solve_wrapper(squares_guesses)
                            if solved_puzzle is not None:
                                process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                                img_result = process.unwarp_image(warped, img_result, corners, solve_time)
                                seen[squares_guesses] = (solved_puzzle, solve_time)
                                if toggles["verbose"]:
                                    print(f"[SUDOKU] Solved in {solve_time}, caching key.")
                            else:
                                seen[squares_guesses] = False
                                if toggles["verbose"]:
                                    print("[SUDOKU] Unsolvable puzzle cached.")
                        except Exception as e:
                            print("[ERR] sudoku solve/draw/unwarp failed:", e)

        # Compose debug overlays
        overlay = img_result.copy()
        if show_help:
            y = 25
            for line in ["DEBUG MODE",
                         "h:help 1:proc 2:warped 3:mask 4:numbers 5:warped_proc g:grid v:verbose s:snap q:quit"]:
                put_banner(overlay, line, y=y, color=(255, 255, 0))
                y += 22

        show_if(toggles["result"], "Result", overlay)
        show_if(toggles["processed"], "1) Processed", proc_u8 if 'proc_u8' in locals() else None)
        show_if(toggles["warped"], "2) Warped", warped_u8 if 'warped_u8' in locals() else None)
        show_if(toggles["mask"], "3) Mask", msk_u8 if 'msk_u8' in locals() else None)
        show_if(toggles["numbers"], "4) Numbers", nums_u8 if 'nums_u8' in locals() else None)
        show_if(toggles["warped_processed"], "5) Warped Processed", wp_u8 if 'wp_u8' in locals() else None)

        # Optionally draw 9x9 grid overlay on warped
        if toggles["squares_grid"] and 'warped' in locals() and warped is not None:
            grid = to_uint8(warped.copy())
            h, w = grid.shape[:2]
            for i in range(1, 9):
                x = int(w * i / 9)
                y = int(h * i / 9)
                cv2.line(grid, (x, 0), (x, h), (0, 255, 0), 1)
                cv2.line(grid, (0, y), (w, y), (0, 255, 0), 1)
            show_if(True, "Grid overlay", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('1'):
            toggles["processed"] = not toggles["processed"]
        elif key == ord('2'):
            toggles["warped"] = not toggles["warped"]
        elif key == ord('3'):
            toggles["mask"] = not toggles["mask"]
        elif key == ord('4'):
            toggles["numbers"] = not toggles["numbers"]
        elif key == ord('5'):
            toggles["warped_processed"] = not toggles["warped_processed"]
        elif key == ord('g'):
            toggles["squares_grid"] = not toggles["squares_grid"]
        elif key == ord('v'):
            toggles["verbose"] = not toggles["verbose"]
            print("[VERBOSE]", toggles["verbose"])
        elif key == ord('s'):
            # Save snapshots
            nonlocal_snap_idx = 0  # not using nonlocal; we just increment a local counter for filenames
            nonlocal_snap_idx += 1
            stamp = f"{int(t.time())}"
            if 'proc_u8' in locals() and proc_u8 is not None:
                cv2.imwrite(str(save_dir / f"{stamp}_processed.png"), proc_u8)
            if 'warped_u8' in locals() and warped_u8 is not None:
                cv2.imwrite(str(save_dir / f"{stamp}_warped.png"), warped_u8)
            if 'wp_u8' in locals() and wp_u8 is not None:
                cv2.imwrite(str(save_dir / f"{stamp}_warped_processed.png"), wp_u8)
            if 'msk_u8' in locals() and msk_u8 is not None:
                cv2.imwrite(str(save_dir / f"{stamp}_mask.png"), msk_u8)
            if 'nums_u8' in locals() and nums_u8 is not None:
                cv2.imwrite(str(save_dir / f"{stamp}_numbers.png"), nums_u8)
            cv2.imwrite(str(save_dir / f"{stamp}_result.png"), to_uint8(img_result))
            print(f"[SNAP] Saved set to {save_dir}")

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
