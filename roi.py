import cv2
import os

# Folders
input_folder = r"/home/nitr/Desktop/Mousumi_Behera/Results/roi_input"
output_folder = r"/home/nitr/Desktop/Mousumi_Behera/Results/roi_output/VV"

os.makedirs(output_folder, exist_ok=True)

# Image list
images = [
    f for f in os.listdir(input_folder)
    if f.lower().strip().endswith(('.png', '.jpg', '.jpeg', ".bmp"))
]
images.sort()

# --------------------------------
# STEP 0: Ask user what to do
# --------------------------------
print("Choose ROI method:")
print("1 - Select ROI once (apply to ALL images)")
print("2 - Use fixed ROIs")
print("3 - Select ROI for EACH image")

choice = input("Enter 1, 2 or 3: ")

# --------------------------------
# STEP 1: ROI Selection (only for option 1)
# --------------------------------
if choice == "1":

    first_img = cv2.imread(os.path.join(input_folder, images[0]))

    print("Select ROIs on FIRST image only.")
    print("Press ENTER after selecting each ROI.")
    print("Press ESC when finished.")

    rois_global = cv2.selectROIs("Select ROIs", first_img)
    cv2.destroyAllWindows()

    print("Saved ROIs:", rois_global)

elif choice == "2":

    rois_global = [
        (325, 90, 135, 62)
    ]

    print("Using predefined ROIs:", rois_global)

elif choice == "3":
    print("You will select ROIs for EACH image.")

else:
    print("Invalid choice. Exiting.")
    exit()

# --------------------------------
# STEP 2: Apply ROIs
# --------------------------------
for name in images:

    img_path = os.path.join(input_folder, name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {name}, cannot read")
        continue

    base = os.path.splitext(name)[0]
    img_boxes = img.copy()

    # --------------------------------
    # ROI selection per image (option 3)
    # --------------------------------
    if choice == "3":
        print(f"\nSelect ROIs for: {name}")
        print("Press ENTER after selecting each ROI.")
        print("Press ESC when finished.")

        rois = cv2.selectROIs("Select ROIs", img)
        cv2.destroyAllWindows()

        print("Saved ROIs:", rois)

    else:
        rois = rois_global

    # --------------------------------
    # Apply ROIs
    # --------------------------------
    for i, (x, y, w, h) in enumerate(rois):

        crop = img[y:y+h, x:x+w]

        cv2.imwrite(
            os.path.join(output_folder, f"{base}_ROI{i+1}.png"),
            crop
        )

        cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Save annotated image
    cv2.imwrite(
        os.path.join(output_folder, f"{base}_withROIs.png"),
        img_boxes
    )

# --------------------------------
# DONE
# --------------------------------
if choice == "1":
    print("✅ SAME ROIs applied to ALL images.")
elif choice == "2":
    print("✅ Fixed ROIs applied to ALL images.")
else:
    print("✅ Individual ROIs applied per image.")