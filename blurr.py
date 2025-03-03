import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Global variables
orig_img = None         # Original image (full resolution)
final_result = None     # Final image (effects applied, full resolution)
preview_img = None      # Result after zoom & pan (same size as original)
display_scale = 1.0     # Scale used to display the image (preserving aspect ratio)
offset_x, offset_y = 0, 0  # Offsets for letterboxing

blur_fields = []        # List of effect fields (each is a list of 4 points in original coords)
active_field = []       # Currently being defined field
adding_mode = False     # True when adding a new field

# For zooming & panning
zoom_factor = 1.0
pan_x, pan_y = 0, 0
is_dragging = False
drag_start = (0, 0)
initial_pan = (0, 0)
click_start = None  # For distinguishing a click from a drag when in adding mode

# Effect mode: "blur" or "pixelate"
effect_mode = "blur"

window_name = "blurr"

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_perspective_effect(img, pts, ksize):
    global effect_mode
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    if effect_mode == "blur":
        effect_warped = cv2.GaussianBlur(warped, (ksize, ksize), 0)
    elif effect_mode == "pixelate":
        pixel_factor = max(1, ksize//10)
        small = cv2.resize(warped, (maxWidth//pixel_factor, maxHeight//pixel_factor), interpolation=cv2.INTER_LINEAR)
        effect_warped = cv2.resize(small, (maxWidth, maxHeight), interpolation=cv2.INTER_NEAREST)
    else:
        effect_warped = warped

    M_inv = cv2.getPerspectiveTransform(dst, rect)
    effect_region = cv2.warpPerspective(effect_warped, M_inv, (img.shape[1], img.shape[0]))
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.fillConvexPoly(mask, rect.astype(int), 255)
    result = img.copy()
    result[mask==255] = effect_region[mask==255]
    return result

def update_preview():
    global preview_img, final_result, display_scale, offset_x, offset_y
    if orig_img is None:
        return
    result = orig_img.copy()
    k_val = cv2.getTrackbarPos("Kernel", window_name)
    if k_val % 2 == 0:
        k_val += 1
    if k_val < 3:
        k_val = 3
    for field in blur_fields:
        result = apply_perspective_effect(result, field, k_val)
    if active_field:
        pts_array = np.array(active_field, dtype=int)
        for p in active_field:
            cv2.circle(result, tuple(p), 5, (0,0,255), -1)
        if len(active_field)>1:
            cv2.polylines(result, [pts_array], False, (0,255,0), 2)
    final_result = result.copy()
    
    rows, cols = result.shape[:2]
    M = np.array([[zoom_factor, 0, pan_x],
                  [0, zoom_factor, pan_y]])
    preview_img = cv2.warpAffine(result, M, (cols, rows))
    
    try:
        winRect = cv2.getWindowImageRect(window_name)
        win_width, win_height = winRect[2], winRect[3]
    except:
        win_width, win_height = cols, rows

    scale_w = win_width / cols
    scale_h = win_height / rows
    display_scale = min(scale_w, scale_h)
    disp_width = int(cols * display_scale)
    disp_height = int(rows * display_scale)
    resized = cv2.resize(preview_img, (disp_width, disp_height))
    
    display_img = np.zeros((win_height, win_width, 3), dtype=np.uint8)
    offset_x = (win_width - disp_width) // 2
    offset_y = (win_height - disp_height) // 2
    display_img[offset_y:offset_y+disp_height, offset_x:offset_x+disp_width] = resized
    cv2.imshow(window_name, display_img)

def mouse_callback(event, x, y, flags, param):
    global active_field, adding_mode, is_dragging, drag_start, initial_pan, click_start
    global pan_x, pan_y, zoom_factor, display_scale, offset_x, offset_y

    # If in adding mode, use left button for point addition.
    if adding_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            click_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            # Determine if this was a click (not a drag).
            if click_start is not None:
                dx = x - click_start[0]
                dy = y - click_start[1]
                if abs(dx) < 5 and abs(dy) < 5:
                    # Convert x,y from window coordinates to original image coordinates.
                    disp_width = int(orig_img.shape[1] * display_scale)
                    disp_height = int(orig_img.shape[0] * display_scale)
                    local_x = x - offset_x
                    local_y = y - offset_y
                    # Clamp to image bounds.
                    local_x = max(0, min(local_x, disp_width-1))
                    local_y = max(0, min(local_y, disp_height-1))
                    orig_x = int((local_x / display_scale - pan_x) / zoom_factor)
                    orig_y = int((local_y / display_scale - pan_y) / zoom_factor)
                    active_field.append([orig_x, orig_y])
                    update_preview()
                    if len(active_field) == 4:
                        blur_fields.append(active_field.copy())
                        active_field.clear()
                        adding_mode = False
                        update_preview()
                click_start = None
            return
        # In adding mode, we do not allow panning.
        return

    # When not in adding mode, use left button for panning.
    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        drag_start = (x, y)
        initial_pan = (pan_x, pan_y)
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        dx = x - drag_start[0]
        dy = y - drag_start[1]
        pan_x = initial_pan[0] + dx / display_scale
        pan_y = initial_pan[1] + dy / display_scale
        update_preview()
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False

    # Scroll-wheel zoom centered on mouse.
    if event == cv2.EVENT_MOUSEWHEEL:
        m_x = x - offset_x
        m_y = y - offset_y
        if m_x < 0 or m_y < 0 or m_x > orig_img.shape[1] * display_scale or m_y > orig_img.shape[0] * display_scale:
            return
        old_zoom = zoom_factor
        if flags > 0:
            zoom_factor += 0.1
        else:
            zoom_factor = max(0.1, zoom_factor - 0.1)
        X = (m_x / display_scale - pan_x) / old_zoom
        Y = (m_y / display_scale - pan_y) / old_zoom
        pan_x = m_x / display_scale - zoom_factor * X
        pan_y = m_y / display_scale - zoom_factor * Y
        update_preview()
        return

def nothing(x):
    update_preview()

def open_image():
    global orig_img, blur_fields, active_field, adding_mode, zoom_factor, pan_x, pan_y
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        orig_img = cv2.imread(file_path)
        blur_fields.clear()
        active_field.clear()
        adding_mode = False
        zoom_factor = 1.0
        pan_x, pan_y = 0, 0
        update_preview()
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)

def save_image():
    global final_result
    if final_result is not None:
        root = tk.Tk()
        root.withdraw()
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        if save_path:
            cv2.imwrite(save_path, final_result)
            print("Saved as", save_path)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)

def main():
    global orig_img, adding_mode, active_field, blur_fields, effect_mode
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 300, 300)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.createTrackbar("Kernel", window_name, 51, 101, nothing)
    
    update_preview()
    print("Controls:")
    print("  o: Open an image")
    print("  a: Add a new field (click 4 points with left button)")
    print("  r: Reset all fields")
    print("  s: Save final image (via save dialog)")
    print("  q: Quit")
    print("  Scroll Wheel: Zoom (centered on mouse)")
    print("  Left-drag: Pan the image (when not adding a field)")
    print("  b: Toggle between blur and pixelate (current mode: {})".format(effect_mode))
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('o'):
            open_image()
        elif key == ord('a'):
            active_field.clear()
            adding_mode = True
            print("Adding a new field. Left-click 4 points.")
            update_preview()
        elif key == ord('r'):
            blur_fields.clear()
            active_field.clear()
            adding_mode = False
            update_preview()
            print("Reset all fields.")
        elif key == ord('s'):
            save_image()
        elif key == ord('b'):
            effect_mode = "pixelate" if effect_mode == "blur" else "blur"
            print("Switched to:", effect_mode)
            update_preview()
        elif key == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
