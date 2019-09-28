import argparse
import base64
import io
import os
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler

import dominate
from dominate import tags as t
from dominate.util import raw
import numpy as np
from PIL import Image

from virtex.config import Config
from virtex.factories import TokenizerFactory, PretrainingDatasetFactory


parser = argparse.ArgumentParser(
    description="Visualize image-caption pairs from an LMDB file."
)
parser.add_argument(
    "--config", default=None,
    help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override", nargs="*", default=[],
    help="""A sequence of key-value pairs specifying certain config arguments
    (with dict-like nesting) using a dot operator.""",
)
parser.add_argument(
    "--output", default=None,
    help="Output path to dump HTML of visualized split. Dumped in /tmp if "
    "this argument is left blank.",
)
parser.add_argument(
    "--port", type=int, default=9090,
    help="Free port on localhost for serving visualized data."
)

# Some constants for bootstrap grid class names.
C3 = "col-12 col-sm-6 col-md-3 col-lg-3 col-xl-3"
C12 = "col-12 col-sm-12 col-md-12 col-lg-12 col-xl-12"


def create_instance_html(instance, tokenizer):
    r"""Create HTML structure for a given an instance from LMDB file."""

    image_id, image = instance["image_id"], instance["image"]
    captions = [tokenizer.decode(instance["caption_tokens"].tolist())]

    # Convert torch tensor (CHW) into a numpy unit8 array in (HWC).
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)

    image_id_tag = t.div(t.b(f"Image ID: {image_id}"), _class=C12)

    # Convert image array to base64 string (UTF-8).
    image = Image.fromarray(image)
    buffered = io.BytesIO()
    image.save(buffered, format="png")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()
    image_b64 = f"data:image/png;base64,{image_b64}"

    # Form image tags and unordered list of associated captions.
    image_tag = t.div(
        t.img(src=image_b64, style="width: 100%; horizontal-align: center"),
        _class=C12,
    )
    # Form an unordered list of captions.
    captions_tag = t.div(t.ul(t.li(c) for c in captions), _class=C12)

    # Combine all tags - Image ID, image and captions. Make it look like a
    # clean card so it's good to look at.
    instance_tag = t.div(
        image_id_tag,
        image_tag,
        captions_tag,
        _class="row",
        style="background-color: #eee; border: 1px solid #000; margin: 10px 0 0 0",
    )
    return instance_tag


if __name__ == "__main__":
    _A = parser.parse_args()

    # Do not perform color normalization here because we are visualizing images.
    _A.config_override.extend(["DATA.IMAGE_TRANSFORM_TRAIN", []])
    _C = Config(_A.config, _A.config_override)

    tokenizer = TokenizerFactory.from_config(_C)
    train_dataset = PretrainingDatasetFactory.from_config(_C, split="train")

    # Get the path of LMDB file we are reading for visualization.
    data_lmdb_path = os.path.join(_C.DATA.ROOT, "serialized_train.lmdb")

    # Prepare an HTML document and start adding stuff in it. We will have four
    # columns, with images stacked in them vertically.
    htmldoc = dominate.document(title=data_lmdb_path)

    with htmldoc.head:
        t.meta(charset="utf-8"),
        t.link(
            rel="stylesheet",
            href="//maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css",
        )
        t.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css?family=Inconsolata:400,700",
        )
        # fmt: off
        t.style(raw("body { font-family: \"Inconsolata\", monospace; }"))
        # fmt: on

    # We have four vertical columns of images and captions.
    image_columns = [t.div(_class=C3) for _ in range(4)]

    # keys: {"image_id", "image", "caption_tokens"}
    for i, instance in enumerate(train_dataset):

        if i > 100:
            break

        # Keep adding images one by one in the columns.
        column_to_add = i % len(image_columns)
        image_columns[column_to_add] += create_instance_html(instance, tokenizer)

    # Single row div to put our columns.
    htmldoc += t.div(
        t.div(t.h2(f"Examples from {data_lmdb_path}"), _class="row"),
        t.div(*image_columns, _class="row"),
        _class="container-fluid",
    )

    # ------------------------------------------------------------------------
    # HTML file serving
    # ------------------------------------------------------------------------
    if _A.output is not None:
        with open(_A.output, "w+") as outfile:
            outfile.write(str(htmldoc))

    tempdir_path = tempfile.mkdtemp()
    with open(os.path.join(tempdir_path, "index.html"), "w") as html_file:
        html_file.write(str(htmldoc))

    # Change to temp directory for serving the visualization html.
    original_dirpath = os.path.realpath(os.curdir)
    os.chdir(tempdir_path)
    httpd = HTTPServer(("", _A.port), SimpleHTTPRequestHandler)

    try:
        print(f"Serving {tempdir_path} on localhost:{_A.port} ...")
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
        os.chdir(original_dirpath)
