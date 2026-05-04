# app.py
import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from constraint import Problem
from torchvision import transforms
import logging
logging.basicConfig(level=logging.INFO)

#  Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = "12345"

#  Cài đặt thư viện YOLO và ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classification model
model_cls = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model_cls.fc = torch.nn.Linear(512, 4)  # có 4 class
model_cls.load_state_dict(torch.load("resnet18_4class_drawing.pth", map_location=device))
model_cls = model_cls.to(device).eval()

# Load YOLO model
model_yolo = YOLO("best_01.pt")

# Danh sách lớp tranh
class_names = ["landscape", "portrait", "space"]
preferred_objects = {
    "landscape": {"tree", "house", "car", "person","sun"},
    "portrait": {"person", "face"},
    "space": {"planet", "rocket", "star", "person"}
}

# Hàm phân loại tranh
def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_cls(img_tensor)
        pred = torch.argmax(output, 1).item()
    return class_names[pred]

#hàm chấm điểm
def score_prediction(detected_boxes, category, class_names_yolo):
    labels = [class_names_yolo[int(d.cls)] for d in detected_boxes]
    label_set = set(labels)
    preferred = preferred_objects.get(category, set())

    matched = list(label_set & preferred)
    unmatched = list(label_set - preferred)

    # Constraint programming
    problem = Problem()
    problem.addVariable("score", range(0, 11))

    def scoring_rule(score):
        match_count = len(matched)
        total = len(preferred)
        if total == 0:
            return score == 0
        ratio = match_count / total

        if ratio >= 1.0:
            return score == 10
        elif ratio >= 0.8:
            return 8 <= score < 10
        elif ratio >= 0.5:
            return 6 <= score < 8
        elif match_count >= 1:
            return 3 < score < 6
        else:
            return score <= 3

    problem.addConstraint(scoring_rule, ["score"])
    result = problem.getSolutions()
    #base_score = max(r["score"] for r in result) if result else 0
    base_score = result[-1]["score"] if result else 0
    # Bonus từ các object không liên quan
    bonus = 0.5 * len(unmatched)
    final_score = min(10, base_score + bonus)

    return round(final_score, 1), matched



# ham sinh nhan xet LLM
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="vistral-7b-chat")  # Thay bằng API key

def generate_feedback(category, labels, matched, score):
    prompt = (
        f"Bức tranh thuộc thể loại '{category}' và có các đối tượng được phát hiện là: {labels}. "
        f"Các đối tượng liên quan đến chủ đề: {matched}. "
        f"Điểm đánh giá tổng thể là {score}/10.\n"
        "Hãy viết một đoạn nhận xét ngắn về bức tranh này như một giáo viên nghệ thuật dành cho trẻ em."
        " đừng nhận xét về màu sắc."
        f"Gợi ý thêm đối tượng liên quan đến chủ đề còn thiếu trong {matched} nếu cần, khuyến khích tích cực."
    )

    try:
        response = client.chat.completions.create(
            model="local-model",  
            messages=[
                {"role": "system", "content": "Bạn là giáo viên mỹ thuật đang đánh giá tranh của học sinh tiểu học."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        feedback = response.choices[0].message.content
        print("AI trả về:", feedback)
        return feedback
    except Exception as e:
        print(f"Lỗi khi gọi LM Studio: {e}")
        return "Không thể tạo nhận xét từ AI lúc này."



#  Hàm xử lý ảnh và trả kết quả
import uuid
import shutil
import time
from flask import redirect, url_for, session

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if uploaded_file:
            # Xóa ảnh cũ trong static và uploads
            for folder in ["uploads", "static"]:
                folder_path = os.path.join(os.getcwd(), folder)
                for f in os.listdir(folder_path):
                    if f.endswith((".jpg", ".jpeg", ".png")):
                        try:
                            os.remove(os.path.join(folder_path, f))
                        except Exception as e:
                            print(f"Error deleting {f} in {folder}: {e}")

            # Tạo tên file duy nhất và lưu
            original_filename = secure_filename(uploaded_file.filename)
            file_ext = os.path.splitext(original_filename)[1]
            unique_filename = str(uuid.uuid4()) + file_ext

            upload_path = os.path.join("uploads", unique_filename)
            uploaded_file.save(upload_path)
            logging.info("Đã lưu ảnh, bắt đầu phân loại")

            try:
                # 1. Phân loại tranh
                category = classify_image(upload_path)

                # 2. Dự đoán object bằng YOLO
                results = model_yolo.predict(upload_path, save=True, conf=0.3)[0]
                result_objects = results.boxes.cls.tolist()
                result_labels = [results.names[int(i)] for i in result_objects]
                detected_boxes = results.boxes

                # 3. Chấm điểm
                score, matched = score_prediction(detected_boxes, category, results.names)
                logging.info("Chấm điểm xong, gọi OpenAI")

                # 4. Gọi AI tạo phản hồi
                feedback = generate_feedback(category, result_labels, matched, score)
                logging.info(f"Feedback AI: {feedback}")
                if not feedback or "Không thể tạo nhận xét" in feedback:
                    feedback = "(AI hiện không phản hồi. Bạn vẫn có thể xem điểm và đối tượng nhé!)"

                # 5. Tìm ảnh kết quả YOLO
                # Lấy tên file gốc không chứa đuôi để phòng trường hợp YOLO tự đổi đuôi thành .jpg
                base_filename = os.path.splitext(os.path.basename(results.path))[0]
                yolo_output_path = None

                # Tăng số vòng lặp lên 100 lần (10 giây) để chắc chắn ảnh đã lưu xong
                for _ in range(100):
                    if os.path.exists(results.save_dir):
                        # Quét thư mục save_dir tìm file có tên khớp với base_filename
                        for file in os.listdir(results.save_dir):
                            if file.startswith(base_filename):
                                yolo_output_path = os.path.join(results.save_dir, file)
                                break

                    # Nếu đã tìm được đường dẫn và file thực sự tồn tại
                    if yolo_output_path and os.path.exists(yolo_output_path):
                        break
                    time.sleep(0.1)
                else:
                    return render_template("index.html", error="Không tìm thấy ảnh kết quả YOLO (đã chờ 10s)")

                # 6. Copy sang static
                # Lấy tên file thực tế (có thể đuôi đã bị YOLO đổi)
                actual_filename = os.path.basename(yolo_output_path)
                static_result_path = os.path.join("static", actual_filename)
                shutil.copy(yolo_output_path, static_result_path)

                # Cập nhật lại biến img_path thành actual_filename thay vì unique_filename cũ
                return render_template("index.html",
                                       category=category,
                                       score=score,
                                       matched=matched,
                                       labels=list(set(result_labels)),
                                       img_path=actual_filename,
                                       feedback=feedback)

                # Trả kết quả trực tiếp, không dùng session để tránh lưu cũ
                return render_template("index.html",
                                       category=category,
                                       score=score,
                                       matched=matched,
                                       labels=list(set(result_labels)),
                                       img_path=unique_filename,
                                       feedback=feedback)

            except Exception as e:
                logging.error(f"Lỗi xử lý: {e}")
                return render_template("index.html", error="Đã xảy ra lỗi trong quá trình xử lý ảnh.")

        else:
            return render_template("index.html", error="Bạn chưa chọn ảnh!")

    # GET request
    return render_template("index.html",
                           category=None,
                           score=None,
                           matched=None,
                           labels=None,
                           img_path=None,
                           feedback=None)



if __name__ == "__main__":
    app.run(debug=True)
