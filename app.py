# from flask import Flask,jsonify
# import numpy as np
# import cv2
# from keras.models import load_model
# from flask_cors import CORS
# from datetime import datetime
# import os
# import threading
# import time

# app = Flask(__name__)

# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# info = {}

# haarcascade = "haarcascade_frontalface_default.xml"
# label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
# print("+"*50, "loadin gmmodel")
# model = load_model(r"model.h5")
# cascade = cv2.CascadeClassifier(haarcascade)

# def delete_file_later(filepath, delay=10):
#     time.sleep(delay)
#     if os.path.exists(filepath):
#         os.remove(filepath)

# @app.route('/emotion_detect', methods=["POST"])
# def emotion_detect():
	
# 	found = False 

# 	cap = cv2.VideoCapture(0)
# 	while not(found):
# 		_, frm = cap.read()
  
# 		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 		image_filename = f"yourpic_{timestamp}.jpg"
# 		cv2.imwrite(os.path.join("static/img", image_filename), frm)
  
# 		gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)

# 		faces = cascade.detectMultiScale(gray, 1.4, 1)

# 		for x,y,w,h in faces:
# 			found = True
# 			roi = gray[y:y+h, x:x+w]
# 			cv2.imwrite("static/face.jpg", roi)

# 	roi = cv2.resize(roi, (48,48))

# 	roi = roi/255.0
	
# 	roi = np.reshape(roi, (1,48,48,1))

# 	prediction = model.predict(roi)

# 	print(prediction)
	
# 	prediction = np.argmax(prediction)
# 	prediction = label_map[prediction]

# 	cap.release()
 
# 	threading.Thread(target=delete_file_later, args=(os.path.join("static/img", image_filename),)).start()
  
# 	link = f"http://127.0.0.1:5000/static/img/{image_filename}"
# 	return jsonify({'result': prediction,'link':link})
	

# # if __name__ == "__main__":
	
# # 	# CORS(app, origins="http://localhost:5173")  #Frontend server linked 
# # 	app.run(debug=True)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=False, host='0.0.0.0', port=port)
 
#----------------------------------------------------------------------------------------------------------------------------- 

from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from keras.models import load_model
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
model = load_model("model.h5")
cascade = cv2.CascadeClassifier(haarcascade)

@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.4, 1)

    for x, y, w, h in faces:
        roi = gray[y:y+h, x:x+w]
        break
    else:
        return jsonify({"result": "No face detected", "link": ""})

    roi = cv2.resize(roi, (48, 48)) / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))

    prediction = model.predict(roi)
    emotion = label_map[np.argmax(prediction)]

    filename = f"static/img/captured_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    if not os.path.exists("static/img"):
        os.makedirs("static/img")
    cv2.imwrite(filename, img)
    
    


    return jsonify({"result": emotion, "link": f"http://127.0.0.1:5000/{filename}"})

if __name__ == "__main__":
    app.run(debug=True)
