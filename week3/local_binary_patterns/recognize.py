# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True, 
	help="path to the tesitng images")
args = vars(ap.parse_args())
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# extract the label from the image path, then update the
	# label and data lists
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
	data.append(hist)
	print(f"Training on: {os.path.basename(imagePath)} -> {label}")

# Print training summary
print(f"\nTraining Summary:")
print(f"Total training images: {len(data)}")
print(f"Unique classes: {set(labels)}")
print(f"Class distribution: {[(label, labels.count(label)) for label in set(labels)]}")

# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42, max_iter=10000)
model.fit(data, labels)
print("Model training completed!\n")

# loop over the testing images
testing_images = list(paths.list_images(args["testing"]))
print(f"Found {len(testing_images)} testing images: {[os.path.basename(img) for img in testing_images]}")

for i, imagePath in enumerate(testing_images):
	print(f"\n--- Processing image {i+1}/{len(testing_images)} ---")
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	if image is None:
		print(f"Could not load image: {imagePath}")
		continue
		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist.reshape(1, -1))
	
	# Get decision function scores for confidence
	decision_scores = model.decision_function(hist.reshape(1, -1))
	
	# print the prediction to console for debugging
	print(f"Image: {os.path.basename(imagePath)}")
	print(f"Prediction: {prediction[0]}")
	print(f"Decision scores: {decision_scores}")
	
	# display the image and the prediction
	cv2.putText(image, str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.putText(image, f"Score: {decision_scores[0]:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)
	cv2.putText(image, f"Image {i+1}/{len(testing_images)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, (255, 0, 0), 2)
	cv2.imshow("Image Classification", image)
	print("Press any key to continue to next image, or 'q' to quit...")
	key = cv2.waitKey(0)
	
	# Allow user to quit with 'q'
	if key == ord('q'):
		print("Quitting...")
		break

# Close all windows when done
cv2.destroyAllWindows()
