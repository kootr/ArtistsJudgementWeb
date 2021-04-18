import os

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import judge_artist_name

UPLOAD_FOLDER = "./static/upload_images"
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(["png", "jpg", "gif", "PNG", "JPG", "GIF"])


@app.route("/")
def index():
    return render_template("index.html")


def allowed_file(file_name):
    return "." in file_name and file_name.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


@app.route("/post", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("File not found")
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            flash("File not found")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
            file.save(file_path)
            (h_indexes, artistname, result_score) = judge_artist_name.main(
                file_path, hdf5_path="./model/artist-model_15.hdf5"
            )
        else:
            return redirect(url_for("index"))

        return render_template(
            "index.html",
            h_indexes=h_indexes,
            artistname=artistname,
            result_score=result_score,
            file_path=file_path,
        )
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run()
