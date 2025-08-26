import io
import json
import threading
import time
import cv2
from flask import Flask, render_template, Response, send_file, send_from_directory
from flask import request
from flask import redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import mysql.connector
import os

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sql#/717171",
    database="parking_db",
    port=3306
)
cursor = db.cursor()

UPLOAD_FOLDER = "web/static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video = cv2.VideoCapture('web/video/loop_car.mp4')

current_frame = None
latest_content = {"new_content": ["En attente de l'IA...", "Pas d'instructions"]}
frame_lock = threading.Lock()
clients = []

@app.route("/add_vehicle", methods=['POST'])
def add_vehicle():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    phone = request.form['phone']
    license_plate = request.form['license_plate']
    arrival = request.form['arrival']
    departure = request.form['departure']

    car_image = request.files['car_image']
    if car_image:
        filename = secure_filename(car_image.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        car_image.save(image_path)
    else:
        image_path = ""

    try:
        cursor.execute("""
            INSERT INTO vehicle_info (first_name, last_name, phone, license_plate, arrival, departure, car_image)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (first_name, last_name, phone, license_plate, arrival, departure, filename))
        db.commit()

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

    return redirect(url_for('main_page'))


@app.route("/", methods=['POST', 'GET'])
def main_page():
    if request.method == 'POST':
        print("sending info...")

    return render_template('main_page.html')


@app.route("/camcrop_page", methods=['POST', 'GET'])
def camcrop_page():
    if request.method == 'POST':
        print("sending info...")

    return render_template('camcrop_page.html')


@app.route("/database_page", methods=['POST', 'GET'])
def database_page():
    cursor.execute("SELECT * FROM vehicle_info")
    vehicles = cursor.fetchall()

    return render_template('database_page.html', vehicles=vehicles)


@app.route("/delete_vehicle", methods=['POST'])
def delete_vehicle():
    vehicle_id = request.form['vehicle_id']
    try:
        cursor.execute("DELETE FROM vehicle_info WHERE id = %s", (vehicle_id,))
        db.commit()
        return redirect(url_for('database_page'))
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/images', methods=['GET'])
def get_images():
    """API endpoint that returns a list of image URLs."""
    image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_urls = [f"/api/images/{image}" for image in image_files]
    return jsonify({"images": image_urls})

@app.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    """API endpoint that serves the image from the folder."""
    return send_from_directory("static/uploads", filename)

@app.route("/api/vfeed", methods=['GET'])
def vfeed():
    def generate():
        global current_frame
        # Get the frame rate of the video
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Error: Could not determine frame rate.")
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
            return

        # Calculate delay between frames
        delay = 0.5 / fps

        while True:
            success, frame = video.read()
            if not success:
                # Reset video to the beginning
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(5)
                continue

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame.")
                continue

            # Store the current frame
            with frame_lock:
                current_frame = jpeg

            # Yield the frame as bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            # Introduce delay to match the frame rate
            time.sleep(delay)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/check_license/<plate>", methods=['GET'])
def check_license(plate):

    cursor.execute("SELECT * FROM vehicle_info WHERE license_plate = %s", (plate,))
    result = cursor.fetchone()
    return jsonify({"result": bool(result)}), 200

@app.route("/api/vframe", methods=['GET'])
def vframe():
    global current_frame
    with frame_lock:
        if current_frame is None:
            return jsonify({"error": "No frame available."}), 400

        # Save the current frame to a bytes buffer
        buffer = io.BytesIO(current_frame)
        buffer.seek(0)

        # Return the image as a file
        return send_file(buffer, mimetype='image/jpeg')

@app.route("/api/state_update", methods=['POST'])
def state_update():
    global latest_content
    try:
        data = request.json
        if not data or 'content' not in data:
            return jsonify({"error": "Invalid request"}), 400

        latest_content = {"new_content": data['content']}
        print("Updated content from AI:", latest_content)

        return jsonify({"message": "Content updated", "new_content": latest_content['new_content']}), 200

    except Exception as e:
        print("Error in state_update:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/stream', methods=['GET'])
def stream():
    def sse():
        while True:
            time.sleep(1)  # Avoids busy loop
            data = json.dumps(latest_content)
            yield f"data: {data}\n\n"

    return Response(sse(), mimetype="text/event-stream")

def event_stream():
    while True:
        time.sleep(1)
        if latest_content:
            data = json.dumps(latest_content)
            yield f"data: {data}\n\n"




if __name__ == '__main__':
    app.run(debug=True, threaded=True)
