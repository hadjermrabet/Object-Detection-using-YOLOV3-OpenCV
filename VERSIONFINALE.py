import base64
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
import threading
import numpy as np
import argparse
import cv2
import mysql.connector

def sendemail(filename):
    mail_content = '''Vous avez une alerte'''
    sender_address = ''
    sender_pass = ''
    receiver_address = ''
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'ALERT MESSAGE'
    message.attach(MIMEText(mail_content, 'plain'))
    attach_file = open("storage/"+datetime.now().strftime('%d-%m-%Y')+"/"+filename, 'rb') # Open the file as binary mode
    payload = MIMEImage(attach_file.read())
    #payload.set_payload((attach_file).read())
    #encoders.encode_base64(payload) #encode the attachment
    #add payload header with filename
    payload.add_header('Content-Decomposition', 'attachment', filename=filename)
    message.attach(payload)
    session = smtplib.SMTP('smtp-mail.outlook.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')

def connect():
    try:
        db = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="",
        password="",
        database="alertes")
        print("connection done !!")
        return db
    except mysql.connector.Error as error:
        print("Failed to connect to database {}".format(error))
        exit(1)

def inserttodb(cm):
    connection = connect()
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO `detected_persons` (`img`) VALUES(%s)",[base64.b64encode(cv2.imencode('.jpg',cm)[1])])
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully")
    except mysql.connector.Error as error:
        print("Failed to connect to database {}".format(error))
        exit(1)
    finally:
        cursor.close()
        connection.close()

def makeDirectory():
    os.makedirs('storage',exist_ok=True)
    os.makedirs('storage/'+datetime.now().strftime("%d-%m-%Y"),exist_ok=True)
    
parser = argparse.ArgumentParser()
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/road.mp4")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

#load yolo
def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	print(layers_names)
	print(net.getUnconnectedOutLayers())
	output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers


def display_blob(blob):
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)

#detection d'objects
def detect_objects(img, net, outputLayers):
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs
#
def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	print(outputs)
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			#print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3 :
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
				print(class_id)
	return boxes, confs, class_ids

#
def draw_labels(boxes, confs, colors, class_ids, classes, img):
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			if(label == "person"):
				croped_img = img[y:y+h,x:x+w]
				inserttodb(croped_img)
				makeDirectory()
				filename = datetime.now().strftime('%H-%M-%S.%f')+'.jpg'
				cv2.imwrite('storage/'+datetime.now().strftime('%d-%m-%Y')+"/"+filename,croped_img)
				threading.Thread(target=sendemail,args=(filename,)).start()
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)



# start video

def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(27)
		if key == 'b':
			break
	cap.release()

	if __name__ == '__main__':
		video_play = args.play_video

		if video_play:
			video_path = args.video_path
			if args.verbose:
				print('Opening ' + video_path + " .... ")
			start_video(video_path)

		cv2.destroyAllWindows()

start_video('0')
