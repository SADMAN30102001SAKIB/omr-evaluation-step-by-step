import cv2

from config import (
    ANSWER_KEYS,
    IMAGE_PATH_1,
    IMAGE_PATH_2,
    IS_NEGATIVE,
    IS_ROLL,
    MARK_PER_QUESTION,
    NEGATIVE_MARK,
    NUM_SETS,
    REAL_ANSWERS,
    REAL_ID_NO,
    REAL_MARKS,
    REAL_SET_NO,
    REGENERATE,
    ROLL_DIGITS,
    TOTAL_QUESTIONS,
)
from lib.area import findArea
from lib.crop import cropIfNeeded
from lib.evaluatePage1 import evaluate1
from lib.evaluatePage2 import evaluate2
from lib.find import findSquares
from lib.imgpre import imagePreprocess
from lib.index import boxIndex
from lib.rollbox import checkRoll
from lib.setbox import checkSet
from lib.split import split_list

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def show_image(window_name, image, scale=0.5):
    """Display image in a resizable window with automatic scaling. Wait for key press before closing."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    cv2.resizeWindow(window_name, int(width * scale), int(height * scale))
    cv2.moveWindow(window_name, 500, 100)
    cv2.imshow(window_name, image)
    print(f"   🖼️  Showing: {window_name} - Press any key to continue...")
    cv2.waitKey(0)
    try:
        cv2.destroyWindow(window_name)
    except:
        pass  # Window already closed by user


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================


def main():
    print("=" * 80)
    print("OMR ELITE - Offline Evaluation System")
    print("=" * 80)

    # Prepare answer lists by splitting into batches
    ansList = []
    for j in range(NUM_SETS):
        ans = ANSWER_KEYS[j]
        sizes = [15, 20, 25, 22, 18]
        ansList.append(split_list(ans, sizes))

    # Adjust negative mark
    negativeMark = abs(NEGATIVE_MARK) if IS_NEGATIVE else 0

    # Determine if we need one or two pages
    if TOTAL_QUESTIONS > 35:
        use_two_pages = True
    else:
        use_two_pages = False

    print(f"\n📄 Processing Mode: {'Two Pages' if use_two_pages else 'Single Page'}")
    print(f"📝 Total Questions: {TOTAL_QUESTIONS}")
    print(f"📚 Number of Sets: {NUM_SETS}")
    print(f"✅ Marks per Question: {MARK_PER_QUESTION}")
    print(f"❌ Negative Marking: {f'-{negativeMark}' if IS_NEGATIVE else 'Disabled'}")
    print(f"🆔 Roll Number Detection: {'Enabled' if IS_ROLL else 'Disabled'}")
    print()

    # ========================================================================
    # PROCESS PAGE 1
    # ========================================================================
    print("=" * 80)
    print("PROCESSING PAGE 1")
    print("=" * 80)

    # Read and preprocess image
    print(f"\n📸 Reading image: {IMAGE_PATH_1}")
    image1 = cv2.imread(IMAGE_PATH_1)
    if image1 is None:
        print(f"❌ Error: Could not read image '{IMAGE_PATH_1}'")
        return

    # Resize to standard dimensions
    print("🔧 Resizing to 960x1280...")
    image1 = cv2.resize(image1, (960, 1280))

    # Show original resized image
    show_image("1. Original Image (Resized)", image1)

    # Crop if needed
    print("✂️  Cropping borders if needed...")
    image1 = cropIfNeeded(image1)

    # Grayscale conversion
    print("⚙️  Converting to grayscale...")
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    show_image("2. Grayscale Image", gray1)

    # Binary thresholding
    print("⚙️  Applying binary thresholding (threshold=125)...")
    _, blackNwhite1 = cv2.threshold(gray1, 125, 255, cv2.THRESH_BINARY)
    show_image("3. Binary Thresholding", blackNwhite1)

    # Gaussian blur
    print("⚙️  Applying Gaussian blur (5x5 kernel)...")
    blur1 = cv2.GaussianBlur(blackNwhite1, (5, 5), 0)
    show_image("4. Gaussian Blur (Noise Reduction)", blur1)

    # Canny edge detection
    print("⚙️  Applying Canny edge detection (50, 150)...")
    edged1 = cv2.Canny(blur1, 50, 150)
    show_image("5. Canny Edge Detection", edged1)

    # Find contours
    print("🔍 Finding contours...")
    contours1, _ = cv2.findContours(
        edged1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"   Found {len(contours1)} contours")

    # Find squares
    print("🔍 Detecting square boxes...")
    squares1 = findSquares(contours1)
    print(f"   Detected {len(squares1)} square boxes")

    # Show contours visualization
    contour_vis1 = image1.copy()
    for i, square in enumerate(squares1):
        for point in square:
            cv2.circle(contour_vis1, tuple(point[0]), 12, (0, 0, 255), -1)
    show_image("6. Detected Square Corners", contour_vis1)

    # Calculate areas
    print("📏 Calculating box areas...")
    areaList1, areaListSort1 = findArea(squares1)

    # Determine expected box count and indices
    totalQuestionsPage1 = min(TOTAL_QUESTIONS, 35)
    q2Index, q1Index, rollIndex, setIndex, markIndex, expectedBox = boxIndex(
        totalQuestionsPage1, IS_ROLL, NUM_SETS
    )

    print(f"   Expected boxes: {expectedBox}, Detected boxes: {len(squares1)}")

    # Validate box count
    if len(squares1) != expectedBox:
        print(f"\n❌ ERROR: Page 1 doesn't match OMR specification!")
        print(f"   Expected {expectedBox} boxes, got {len(squares1)} boxes")
        print("\n💡 Suggestions:")
        print("   (i) Ensure this is the correct OMR sheet")
        print("   (ii) Try changing camera angle, keep camera straight")
        print("   (iii) Improve lighting/quality of the picture")
        cv2.destroyAllWindows()
        return

    print("✅ Box count validation passed")

    # Extract roll number if enabled
    idno = -1
    if IS_ROLL:
        print("\n🆔 Extracting roll number...")
        marked_index_roll, x, y, w, h = checkRoll(
            ROLL_DIGITS,
            image1,
            blur1,
            areaList1,
            areaListSort1,
            squares1,
            rollIndex,
            REGENERATE,
            REAL_ID_NO,
        )

        roll = ""
        for i in marked_index_roll:
            if i == -1:
                break
            roll += str(i)

        if len(roll) == ROLL_DIGITS:
            idno = roll
            print(f"   ✅ Roll Number Detected: {idno}")
        else:
            idno = -2
            print(f"   ❌ Could not detect complete {ROLL_DIGITS}-digit roll number")
            print(f"   Detected: {roll if roll else 'None'}")

    # Detect set number if multiple sets
    setno = 1
    tempSet = 1
    if NUM_SETS > 1:
        print("\n📋 Detecting marked set...")
        marked_circle_index, x, y, w, h = checkSet(
            image1,
            blur1,
            areaList1,
            areaListSort1,
            squares1,
            setIndex,
            NUM_SETS,
            REAL_SET_NO,
            REGENERATE,
        )

        if len(marked_circle_index) > 1:
            print(f"   ⚠️  Multiple sets marked!")
            setno = 0
            tempSet = 0
        elif len(marked_circle_index) == 0:
            print(f"   ⚠️  No set marked!")
            setno = 0
            tempSet = 0
        else:
            setno = marked_circle_index[0] + 1
            print(f"   ✅ Set Detected: {setno}")

    if REGENERATE and setno != -1:
        setno = REAL_SET_NO

    # Handle no set detected case
    if setno == 0:
        setno = 1
        tempSet = 0
        # Create dummy answer list
        ansList = []
        for j in range(NUM_SETS):
            ans = []
            for i in range(TOTAL_QUESTIONS):
                ans.append("-1")
            ansList.append(split_list(ans, [15, 20, 25, 22, 18]))

    # ========================================================================
    # EVALUATE QUESTIONS
    # ========================================================================

    marked_index = []

    if use_two_pages:
        # Process Page 2 first
        print("\n" + "=" * 80)
        print("PROCESSING PAGE 2")
        print("=" * 80)

        if not IMAGE_PATH_2:
            print("❌ Error: Page 2 image path not provided but questions > 35")
            cv2.destroyAllWindows()
            return

        print(f"\n📸 Reading image: {IMAGE_PATH_2}")
        image2_orig = cv2.imread(IMAGE_PATH_2)
        if image2_orig is None:
            print(f"❌ Error: Could not read image '{IMAGE_PATH_2}'")
            cv2.destroyAllWindows()
            return

        # Preprocess page 2
        image2, blur2, contours2 = imagePreprocess(IMAGE_PATH_2)

        # Show page 2 processing stages
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        _, blackNwhite2 = cv2.threshold(gray2, 125, 255, cv2.THRESH_BINARY)
        blur2_vis = cv2.GaussianBlur(blackNwhite2, (5, 5), 0)
        edged2 = cv2.Canny(blur2_vis, 50, 150)

        show_image("7. Page 2 - Grayscale", gray2)
        show_image("8. Page 2 - Binary Thresholding", blackNwhite2)
        show_image("9. Page 2 - Gaussian Blur", blur2_vis)
        show_image("10. Page 2 - Canny Edge Detection", edged2)

        totalQuestionsPage2 = TOTAL_QUESTIONS - 35
        box = 1
        if totalQuestionsPage2 > 25:
            box += 1
            if totalQuestionsPage2 > 47:
                box += 1

        squares2 = findSquares(contours2)
        print(f"   Detected {len(squares2)} square boxes (expected {box})")

        # Show page 2 contours
        contour_vis2 = image2.copy()
        for square in squares2:
            for point in square:
                cv2.circle(contour_vis2, tuple(point[0]), 12, (0, 0, 255), -1)
        show_image("11. Page 2 - Detected Square Corners", contour_vis2)

        if len(squares2) != box:
            print(f"\n❌ ERROR: Page 2 doesn't match OMR specification!")
            print(f"   Expected {box} boxes, got {len(squares2)} boxes")
            cv2.destroyAllWindows()
            return

        areaList2, areaListSort2 = findArea(squares2)

        # Evaluate page 2
        print("\n📝 Evaluating Page 2 answers...")
        marks2, f, marked_index2 = evaluate2(
            IMAGE_PATH_2,
            TOTAL_QUESTIONS,
            MARK_PER_QUESTION,
            IS_NEGATIVE,
            negativeMark,
            ansList,
            "memory_output2.jpg",
            setno,
            REGENERATE,
            REAL_ANSWERS,
        )

        if f == -1:
            print(f"❌ Error in Page 2: {marks2}")
            cv2.destroyAllWindows()
            return

        print(f"   Page 2 Marks: {marks2}")
        marked_index = marked_index2

        # Now evaluate page 1
        print("\n📝 Evaluating Page 1 answers...")
        marks1, marked_index1 = evaluate1(
            35,
            MARK_PER_QUESTION,
            IS_NEGATIVE,
            negativeMark,
            ansList,
            "memory_output1.jpg",
            marks2,
            image1,
            blur1,
            areaList1,
            areaListSort1,
            squares1,
            q1Index,
            q2Index,
            markIndex,
            setno,
            REGENERATE,
            REAL_ANSWERS,
            REAL_MARKS,
        )

        print(f"   Page 1 Marks: {marks1}")
        marked_index = marked_index1 + marked_index

        total_marks = marks1 + marks2

        # Show final marked images
        # Read the evaluated images from memory (they were saved temporarily)
        final_image1 = image1  # Already modified in-place by evaluate1
        final_image2 = cv2.imread("memory_output2.jpg")

        show_image("12. Page 1 - Final Evaluated (with marks)", final_image1)
        show_image("13. Page 2 - Final Evaluated", final_image2)

    else:
        # Single page evaluation
        print("\n📝 Evaluating answers...")
        marks1, marked_index = evaluate1(
            TOTAL_QUESTIONS,
            MARK_PER_QUESTION,
            IS_NEGATIVE,
            negativeMark,
            ansList,
            "memory_output1.jpg",
            0,
            image1,
            blur1,
            areaList1,
            areaListSort1,
            squares1,
            q1Index,
            q2Index,
            markIndex,
            setno,
            REGENERATE,
            REAL_ANSWERS,
            REAL_MARKS,
        )

        total_marks = marks1

        # Show final marked image
        show_image("7. Final Evaluated Image (with marks)", image1)

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    if IS_ROLL:
        if idno == -2:
            print(f"🆔 Roll Number: ❌ Could not detect {ROLL_DIGITS}-digit ID")
        else:
            print(f"🆔 Roll Number: {idno}")

    if NUM_SETS > 1:
        if tempSet == 0 and setno == 1:
            print(f"📋 Set Number: ⚠️  Not detected (defaulted to Set 1)")
        else:
            print(f"📋 Set Number: {setno}")

    print(f"📊 Total Marks: {total_marks}/{TOTAL_QUESTIONS * MARK_PER_QUESTION}")
    print(
        f"📈 Percentage: {(total_marks / (TOTAL_QUESTIONS * MARK_PER_QUESTION) * 100):.2f}%"
    )
    print(f"\n📝 Marked Indices: {marked_index}")

    # Print question-wise results
    print("\n" + "-" * 80)
    print("Question-wise Analysis:")
    print("-" * 80)
    for i, marked in enumerate(marked_index):
        q_num = i + 1
        if marked == 0:
            print(f"Q{q_num:3d}: Not marked")
        else:
            print(f"Q{q_num:3d}: Marked option(s) = {marked}")

    print("\n" + "=" * 80)
    print("✅ Processing Complete!")
    print("=" * 80)

    # Clean up temporary files
    import os
    import shutil

    temp_files = ["memory_output1.jpg", "memory_output2.jpg"]
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass

    # Remove __pycache__ folders
    base_dir = os.path.dirname(__file__)
    pycache_paths = [
        os.path.join(base_dir, "__pycache__"),
        os.path.join(base_dir, "lib", "__pycache__"),
    ]
    for pycache_path in pycache_paths:
        try:
            if os.path.exists(pycache_path):
                shutil.rmtree(pycache_path)
        except:
            pass

    print("\n👋 Exiting...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback

        traceback.print_exc()
        cv2.destroyAllWindows()

        # Clean up temporary files on error too
        import os
        import shutil

        temp_files = ["memory_output1.jpg", "memory_output2.jpg"]
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        # Remove __pycache__ folders
        base_dir = os.path.dirname(__file__)
        pycache_paths = [
            os.path.join(base_dir, "__pycache__"),
            os.path.join(base_dir, "lib", "__pycache__"),
        ]
        for pycache_path in pycache_paths:
            try:
                if os.path.exists(pycache_path):
                    shutil.rmtree(pycache_path)
            except:
                pass

        cv2.waitKey(0)
        cv2.destroyAllWindows()
