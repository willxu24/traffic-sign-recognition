import os
from xml.dom.minidom import Document

import cv2


def write_img_to_xml(imgfile, xmlfile):
    img = cv2.imread(imgfile)
    img_folder, img_name = os.path.split(imgfile)
    img_height, img_width, img_depth = img.shape
    doc = Document()

    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)

    folder = doc.createElement("folder")
    folder.appendChild(doc.createTextNode(img_folder))
    annotation.appendChild(folder)

    filename = doc.createElement("filename")
    filename.appendChild(doc.createTextNode(img_name))
    annotation.appendChild(filename)

    size = doc.createElement("size")
    annotation.appendChild(size)

    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(img_width)))
    size.appendChild(width)

    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(img_height)))
    size.appendChild(height)

    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str(img_depth)))
    size.appendChild(depth)

    with open(xmlfile, "w") as f:
        doc.writexml(f, indent="\t", addindent="\t", newl="\n", encoding="utf-8")


def write_imgs_to_xmls(imgdir, xmldir):
    img_names = os.listdir(imgdir)
    for img_name in img_names:
        img_file = os.path.join(imgdir, img_name)
        xml_file = os.path.join(xmldir, img_name.split(".")[0] + ".xml")
        print(img_name, "has been written to xml file in ", xml_file)
        write_img_to_xml(img_file, xml_file)


if __name__ == "__main__":
    img_folder = "./data/a_cropped_images"
    xml_folder = "./data/b_image_xmls"
    write_imgs_to_xmls(img_folder, xml_folder)
