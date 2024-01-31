import os
import sys
import smtplib

from models.DL.InceptionResNetV2Model import InceptionResNetV2Model
from models.DL.AlexNetModel import AlexNetModel
from models.DL.ResNet50Model import ResNet50Model

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

APP_PASSWORD = '' # REMOVE LATER ******************************************************

def run_all_models(directories_list):
    for dirs in directories_list:
        base_dir, base_rgb_dir, base_disease_dir, base_leaf_dir = dirs

        # Base output directory
        last_part = os.path.basename(base_dir)
        if not os.path.exists('outputs' + last_part):
            os.makedirs('outputs' + last_part)
        base_output_dir = 'outputs' + last_part

        # Run ResNet50 Model
        resnet50_output_dir = os.path.join(base_output_dir, 'ResNet50')
        resnet50_model = ResNet50Model(base_rgb_dir, base_disease_dir, base_leaf_dir)
        resnet50_history = resnet50_model.compile_and_train(epochs=10, batch_size=32, output_dir=resnet50_output_dir)
        
        # Run AlexNet Model
        alexnet_output_dir = os.path.join(base_output_dir, 'AlexNet')
        alexnet_model = AlexNetModel(base_rgb_dir, base_disease_dir, base_leaf_dir)
        alexnet_history = alexnet_model.compile_and_train(epochs=10, batch_size=32, output_dir=alexnet_output_dir)

        # Run Inception ResNet V2 Model
        inception_resnet_v2_output_dir = os.path.join(base_output_dir, 'InceptionResNetV2')
        inception_resnet_v2_model = InceptionResNetV2Model(base_rgb_dir, base_disease_dir, base_leaf_dir)
        inception_resnet_v2_history = inception_resnet_v2_model.compile_and_train(epochs=10, batch_size=32, output_dir=inception_resnet_v2_output_dir)

        # Send email notification for each model
        send_email(
            "Model Training Complete for " + last_part,
            "The training of the model with base directory " + last_part + " has been completed successfully.",
            "jhaenel@siue.edu",
            "josephhaenel@gmail.com"
        )
    
def send_email(subject, message, recipient_email, sender_email):
    # ... [Email sending code remains the same]

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python DLrun.py <base_directory1> <RGB_dir1> <GT_Disease_dir1> <GT_Leaf_dir1> [more directories]")
        sys.exit(1)

    # Group arguments into sets of four
    args = sys.argv[1:]
    if len(args) % 4 != 0:
        print("Error: Each model requires four directories. Please check your inputs.")
        sys.exit(1)

    directories_list = [args[i:i+4] for i in range(0, len(args), 4)]
    run_all_models(directories_list)
