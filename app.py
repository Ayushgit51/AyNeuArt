import os
import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_wtf.file import FileField
from wtforms import SubmitField, FloatField, HiddenField
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms

# Import your model code
from utils.models import VGGEncoder, Decoder
from utils.utils import ImageFolderDataset


# ---------------- APP SETUP ---------------- #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ---------------- FORM ---------------- #
class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


# ---------------- DEVICE ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- MODEL LOAD ---------------- #
encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        r"C:\Users\goodb\OneDrive\Desktop\NST Code\experiment\final_exp\decoder_final.pth",
        map_location=device
    )
)

encoder.eval()
decoder.eval()


# ---------------- HELPERS ---------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_image(image, path):
    image = image.detach().cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    content_image = transform(content_image).unsqueeze(0).to(device)
    style_image = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        t = ImageFolderDataset.adaptive_instance_normalization(
            content_feats,
            style_feats
        )

        t = alpha * t + (1 - alpha) * content_feats
        output = decoder(t)

    return output


# ---------------- ROUTES ---------------- #
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':

        # -------- CONTENT IMAGE -------- #
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                form.content.data.save(content_path)
        else:
            content_filename = form.content_path.data

        # -------- STYLE IMAGE -------- #
        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                form.style.data.save(style_path)
        else:
            style_filename = form.style_path.data

        # -------- CHECK -------- #
        if not content_filename:
            error = "Please upload content image"
        if not style_filename:
            error = "Please upload style image"

        # -------- STYLE TRANSFER -------- #
        if content_filename and style_filename and not error:

            try:
                content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                alpha = float(form.alpha.data)

                output = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                result_filename = "stylized_" + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

                save_image(output, result_path)

                result_image = result_filename

            except Exception as e:
                error = str(e)

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ---------------- STATIC FILES ---------------- #
@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


# ---------------- RUN ---------------- #
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
