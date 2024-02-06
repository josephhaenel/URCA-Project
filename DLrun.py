import os
import sys
import smtplib

from models.DL.InceptionResNetV2Model import InceptionResNetV2Model
from models.DL.AlexNetModel import AlexNetModel
from models.DL.ResNet50Model import ResNet50Model

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

APP_PASSWORD = '' # REMOVE LATER ******************************************************


def run_all_models(base_directories):
    for base_dir in base_directories:
        base_rgb_dir = os.path.join(base_dir, 'RGB')
        base_disease_dir = os.path.join(base_dir, 'GroundTruth_Disease')
        base_leaf_dir = os.path.join(base_dir, 'GroundTruth_Leaf')
        
        # Base output directory
        last_part = os.path.basename(base_dir)
        number = 1
        while True:
            if not os.path.exists(os.path.join('outputs', str(number), 'outputs' + last_part)):
                os.makedirs(os.path.join('outputs', str(number), 'outputs' + last_part))
                break
            number += 1    
        base_output_dir = os.path.join('outputs', str(number), 'outputs' + last_part)
        
        learning_rate = 0.001 # 0.001, 0.0001
        val_split = 0.9 # 0.9, 0.7, 0.5, 0.3, 0.1
        
        infoPath = os.path.join('outputs', str(number), 'info.txt')
            
        with open(infoPath, "w") as f:
            # Write learning rate and validation split to info.txt file
            f.write("learningRate = " + str(learning_rate) + "\n" + "valSplit = " + str(val_split))
            
        
        # Run ResNet50 Model
        # resnet50_output_dir = os.path.join(base_output_dir, 'ResNet50')
        # resnet50_model = ResNet50Model(base_rgb_dir, base_disease_dir, base_leaf_dir, learning_rate, val_split)
        # resnet50_history = resnet50_model.compile_and_train(epochs=10, batch_size=32, output_dir=resnet50_output_dir)
        
                # Run AlexNet Model
        alexnet_output_dir = os.path.join(base_output_dir, 'AlexNet')
        alexnet_model = AlexNetModel(base_rgb_dir, base_disease_dir, base_leaf_dir, learning_rate, val_split)
        alexnet_history = alexnet_model.compile_and_train(epochs=10, batch_size=32, output_dir=alexnet_output_dir)
        
            # Run Inception ResNet V2 Model
        # inception_resnet_v2_output_dir = os.path.join(base_output_dir, 'InceptionResNetV2')
        # inception_resnet_v2_model = InceptionResNetV2Model(base_rgb_dir, base_disease_dir, base_leaf_dir, learning_rate, val_split)
        # inception_resnet_v2_history = inception_resnet_v2_model.compile_and_train(epochs=10, batch_size=32, output_dir=inception_resnet_v2_output_dir)
    
    
    # send_email(
    #     "Model Training Complete",
    #     "The training of all models has been completed successfully.",
    #     "jhaenel@siue.edu",
    #     "josephhaenel@gmail.com"
    # )
    
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
    if len(sys.argv) < 1:
        print("Usage: python DLrun.py <base_directory1> [<base_directory2> ...]")
        sys.exit(1)

    base_directories = sys.argv[1:]
    run_all_models(base_directories)
