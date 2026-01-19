import cv2
import numpy as np
import argparse

# State
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point selected: {x}, {y}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to image')
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.source)
    if img is None:
        print("Error: Could not load image.")
        return

    # Resize if too large
    h, w = img.shape[:2]
    if w > 1280:
        scale = 1280 / w
        img = cv2.resize(img, (1280, int(h * scale)))

    # Clone for display
    display_img = img.copy()

    print("\n--- Bird's Eye View Tool ---")
    print("INSTRUCTIONS:")
    print("1. Click 4 points on the ROAD making a TRAPEZOID shape.")
    print("   (Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left)")
    print("2. Press ANY KEY to transform.")
    
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", mouse_callback)

    while True:
        temp_img = display_img.copy()
        for p in points:
            cv2.circle(temp_img, tuple(p), 5, (0, 0, 255), -1)
        
        # Draw lines connecting points
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(temp_img, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 2)
        if len(points) == 4:
             cv2.line(temp_img, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

        cv2.imshow("Select Points", temp_img)
        
        key = cv2.waitKey(1)
        if len(points) == 4:
            print("4 points selected. Press standard key to wrap...")
            cv2.waitKey(0) # Wait for confirmation
            break

    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Exited without selecting 4 points.")
        return

    # 1. Source Points (The 4 points you clicked)
    src_pts = np.float32(points)

    # 2. Destination Points (Top-Down view)
    # Mapping trapezoid to 400x600 rectangle
    w_out, h_out = 400, 600
    dst_pts = np.float32([
        [100, h_out],       # Bottom-Left
        [w_out-100, h_out], # Bottom-Right
        [w_out-100, 0],     # Top-Right
        [100, 0]            # Top-Left
    ])

    # 3. Calculate Homography Matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. Warp Perspective
    # We use a larger width to catch the surroundings
    warped_img = cv2.warpPerspective(img, M, (w_out, h_out))

    # Show results
    cv2.imshow("Original", img)
    cv2.imshow("Bird's Eye View", warped_img)
    
    # Save
    cv2.imwrite("output/birds_eye_result.jpg", warped_img)
    print("Saved result to output/birds_eye_result.jpg")
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
