import argparse
import os
import glob
import pydicom
import numpy as np
import mrcnn.model as modellib

from mrcnn import utils
from config import from_config_file
from tqdm import tqdm


# Make predictions on test images, write out sample submission
def predict(image_fps, config, min_conf, filepath):
    with open(filepath, 'w') as file:
      for image_id in tqdm(image_fps):
        ds = pydicom.read_file(image_id)
        image = ds.pixel_array
        resize_factor = image.shape[0]

        # Convert to RGB for consistency.
        image = np.stack((image,) * 3, -1)
        image, _, _, _, _ = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)

        patient_id = os.path.splitext(os.path.basename(image_id))[0]

        results = model.detect([image])
        r = results[0]

        out_str = ""
        out_str += patient_id
        out_str += ","
        assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
        if len(r['rois']) == 0:
            pass
        else:
            num_instances = len(r['rois'])

            for i in range(num_instances):
                if r['scores'][i] > min_conf:
                    out_str += ' '
                    out_str += str(round(r['scores'][i], 2))
                    out_str += ' '

                    # x1, y1, width, height
                    x1 = r['rois'][i][1]
                    y1 = r['rois'][i][0]
                    width = r['rois'][i][3] - x1
                    height = r['rois'][i][2] - y1
                    bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                       width*resize_factor, height*resize_factor)
                    out_str += bboxes_str

        file.write(out_str+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('threshold', help='min threshold for detected boxes', type=float)
    parser.add_argument('dicom_dir', help='path to directory with eval dicom files')
    parser.add_argument('model', help='path to the pretrained model')
    parser.add_argument('eval_dir', help='path to directory with evaluated images')
    parser.add_argument('--label-file', help='path to label file')
    args = parser.parse_args()

    # Set inference config
    config = from_config_file(args.config)
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    config.__init__()

    # Recreate the model in inference mode
    print('Loading weights from ' + args.model)
    model = modellib.MaskRCNN(
        mode='inference',
        config=config,
        model_dir=os.path.dirname(args.model)
    )
    model.load_weights(args.model, by_name=True)
    print('Weights loaded')

    # Get filenames of test dataset DICOM images
    test_image_fps = glob.glob(os.path.join(args.dicom_dir, '*.dcm'))

    predict(test_image_fps, config, args.threshold, args.label_file)
