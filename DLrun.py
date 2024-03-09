# python DLrun.py /home/studio-lab-user/myNotebook/URCA-Project/Final-URCA-DATASET/highlight_tomato /home/studio-lab-user/myNotebook/URCA-Project/Final-URCA-DATASET/lowlight_tomato /home/studio-lab-user/myNotebook/URCA-Project/Final-URCA-DATASET/Rice_Leaves

import os
import sys
import tensorflow as tf

from models.DL.InceptionResNetV2Model import InceptionResNetV2Model
from models.DL.AlexNetModel import AlexNetModel
from models.DL.ResNet50Model import ResNet50Model

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def run_all_models(base_directories):
    for base_dir in base_directories:
        base_rgb_dir = os.path.join(base_dir, 'RGB')
        base_disease_dir = os.path.join(base_dir, 'GroundTruth_Disease')
        base_leaf_dir = os.path.join(base_dir, 'GroundTruth_Leaf')
        
        # Base output directory
        last_part = os.path.basename(base_dir)
        number = 1
        while True:
            if not os.path.exists(os.path.join('DL_outputs', str(number), 'outputs' + last_part)):
                os.makedirs(os.path.join('DL_outputs', str(number), 'outputs' + last_part))
                break
            number += 1    
        base_output_dir = os.path.join('DL_outputs', str(number), 'outputs' + last_part)
        
        infoPath = os.path.join('DL_outputs', str(number), 'info.txt')
        
        tf.keras.backend.clear_session()
        
        # try:
        #     AlexNet_learning_rate = 0.0001 # 0.00001, 0.0000001
        #     AlexNet_val_split = 0.2 # 0.9, 0.7, 0.5, 0.3, 0.1
            
        #     # Run AlexNet Model
        #     alexnet_output_dir = os.path.join(base_output_dir, 'AlexNet')
        #     alexnet_model = AlexNetModel(base_rgb_dir, base_disease_dir, base_leaf_dir, AlexNet_learning_rate, AlexNet_val_split, last_part)
        #     alexnet_history = alexnet_model.compile_and_train(epochs=30, batch_size=32, output_dir=alexnet_output_dir)
        # except Exception as e:
        #     print(f"Failed to run AlexNet model: {e}")
        
        tf.keras.backend.clear_session()
        
        try:
            InceptionResNetV2_learning_rate = 0.0001 # 0.001, 0.0001
            InceptionResNetV2_val_split = 0.2 # 0.9, 0.7, 0.5, 0.3, 0.1
            
            # Run Inception ResNet V2 Model
            inception_resnet_v2_output_dir = os.path.join(base_output_dir, 'InceptionResNetV2')
            inception_resnet_v2_model = InceptionResNetV2Model(base_rgb_dir, base_disease_dir, base_leaf_dir, InceptionResNetV2_learning_rate, InceptionResNetV2_val_split, last_part)
            inception_resnet_v2_history = inception_resnet_v2_model.compile_and_train(epochs=50, batch_size=32, output_dir=inception_resnet_v2_output_dir)
        except Exception as e:
            print(f"Failed to run InceptionResNetV2 model: {e}")
            
        tf.keras.backend.clear_session()
            
            
        # try:
        #     ResNet50_learning_rate = 0.0001 # 0.001, 0.0001
        #     ResNet50_val_split = 0.2 # 0.9, 0.7, 0.5, 0.3, 0.1   
            
        #     # Run ResNet50 Model
        #     resnet50_output_dir = os.path.join(base_output_dir, 'ResNet50')
        #     resnet50_model = ResNet50Model(base_rgb_dir, base_disease_dir, base_leaf_dir, ResNet50_learning_rate, ResNet50_val_split, last_part)
        #     resnet50_history = resnet50_model.compile_and_train(epochs=100, batch_size=32, output_dir=resnet50_output_dir)
        # except Exception as e:
        #     print(f"Failed to run ResNet50 model: {e}")
        
        tf.keras.backend.clear_session()
        
        # with open(infoPath, "w") as f:
        #     # Write learning rate and validation split to info.txt file
        #     f.write("ResNet50 learningRate = " + str(ResNet50_learning_rate) + "\n" + "ResNet50 valSplit = " + str(ResNet50_val_split))
        #     f.write("AlexNet learningRate = " + str(AlexNet_learning_rate) + "\n" + "AlexNet valSplit = " + str(AlexNet_val_split))
        #     f.write("InceptionResNetV2 learningRate = " + str(InceptionResNetV2_learning_rate) + "\n" + "InceptionResNetV2 valSplit = " + str(InceptionResNetV2_val_split))
    
    
    # send_email(
    #     "Model Training Complete",
    #     "The training of all models has been completed successfully.",
    #     "jhaenel@siue.edu",
    #     "josephhaenel@gmail.com"
    # )
    
# def send_email(subject, message, recipient_email, sender_email):
#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = recipient_email
#     msg['Subject'] = subject

#     msg.attach(MIMEText(message, 'plain'))

#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()
#         server.login(sender_email, APP_PASSWORD)
#         text = msg.as_string()
#         server.sendmail(sender_email, recipient_email, text)
#         server.quit()
#         print("Email sent successfully")
#     except Exception as e:
#         print(f"Failed to send email: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("Usage: python DLrun.py <base_directory1> [<base_directory2> ...]")
        sys.exit(1)

    base_directories = sys.argv[1:]
    run_all_models(base_directories)
