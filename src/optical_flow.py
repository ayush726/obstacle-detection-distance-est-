import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to VIDEO file (mp4, mov, avi)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.source}")
        return

    # Read the first frame
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Video is empty.")
        return

    # Convert to grayscale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # HSV for visualization (Hue=Direction, Value=Speed)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    print("\n--- Optical Flow (Farneback Method) ---")
    print("Shows MOVEMENT (pixels moving).")
    print("Press 'q' to Quit.")

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow to Polar coordinates (Angle + Magnitude)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Set image Hue according to the optical flow direction
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Set image Value according to the optical flow magnitude (speed)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert back to BGR for display
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show Side-by-Side (Original + Flow)
        # Resize to fit screen if needed
        scale = 0.5
        small_frame = cv2.resize(frame2, (0,0), fx=scale, fy=scale)
        small_rgb = cv2.resize(rgb, (0,0), fx=scale, fy=scale)
        combined = np.hstack((small_frame, small_rgb))

        cv2.imshow('Optical Flow (Left: Original, Right: Flow)', combined)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break

        prvs = next_frame

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
