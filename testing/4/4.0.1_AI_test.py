print("Importing")
from inference import get_model
import supervision as sv
import cv2, os

print("Loading model")
# load a pre-trained yolov8n model
model = get_model(model_id="pencil-o4fwn-bnv9f/1")

# Do the file names
filenames = [f for f in os.listdir("images") if os.path.isfile(os.path.join("images", f))]

print("Reading photo")
# define the image url to use for inference
i=0
for path in filenames:
    i+=1
    image_file = f"images/{path}"
    image = cv2.imread(image_file)

    print("Predicting")
    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(image,confidence=0.3)[0]

    print("Getting reuslts")
    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)

    print("Drawing bounding boxes")
    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    print("Displaying image")
    # display the image
    with sv.ImageSink(target_dir_path='./prediction_images') as sink:
        sink.save_image(annotated_image,image_name=f'image{i}.jpg')