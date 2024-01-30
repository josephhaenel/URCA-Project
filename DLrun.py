import os
import sys
import smtplib

from models.DL.InceptionResNetV2Model import InceptionResNetV2Model
from models.DL.AlexNetModel import AlexNetModel
from models.DL.ResNet50Model import ResNet50Model

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

APP_PASSWORD = '' # REMOVE LATER ******************************************************


def run_all_models(base_dir, base_rgb_dir, base_disease_dir, base_leaf_dir):
    # Base output directory
    last_part = os.path.basename(base_dir)
    if not os.path.exists('outputs' + last_part):
        os.makedirs('outputs' + last_part)
    base_output_dir = 'outputs' + last_part
    
            # Run AlexNet Model
    alexnet_output_dir = os.path.join(base_output_dir, 'AlexNet')
    alexnet_model = AlexNetModel(base_rgb_dir, base_disease_dir, base_leaf_dir)
    alexnet_history = alexnet_model.compile_and_train(epochs=10, batch_size=32, output_dir=alexnet_output_dir)
    
        # Run Inception ResNet V2 Model
    inception_resnet_v2_output_dir = os.path.join(base_output_dir, 'InceptionResNetV2')
    inception_resnet_v2_model = InceptionResNetV2Model(base_rgb_dir, base_disease_dir, base_leaf_dir)
    inception_resnet_v2_history = inception_resnet_v2_model.compile_and_train(epochs=10, batch_size=32, output_dir=inception_resnet_v2_output_dir)
    
    
    # Run ResNet50 Model
    resnet50_output_dir = os.path.join(base_output_dir, 'ResNet50')
    resnet50_model = ResNet50Model(base_rgb_dir, base_disease_dir, base_leaf_dir)
    resnet50_history = resnet50_model.compile_and_train(epochs=10, batch_size=32, output_dir=resnet50_output_dir)
    
    send_email(
        "Model Training Complete",
        "The training of all models has been completed successfully.",
        "jhaenel@siue.edu",
        "josephhaenel@gmail.com"
    )
    
def send_email(subject, message, recipient_email, sender_email):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, APP_PASSWORD)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python DLrun.py <base_directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    base_rgb_dir = os.path.join(base_directory, 'RGB')
    base_disease_dir = os.path.join(base_directory, 'GroundTruth_Disease')
    base_leaf_dir = os.path.join(base_directory, 'GroundTruth_Leaf')

    run_all_models(base_directory, base_rgb_dir, base_disease_dir, base_leaf_dir)
