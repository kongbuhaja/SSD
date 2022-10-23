"""custom_dataset dataset."""

import tensorflow_datasets as tfds
from tensorflow_datasets.core.features import BBoxFeature
import xmltodict
from PIL import Image
import numpy as np
import tensorflow as tf

# TODO(custom_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(custom_dataset): BibTeX citation
_CITATION = """
"""


class CustomDataset(tfds.core.GeneratorBasedBuilder):
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
   data.zip files should be located at /root/tensorflow_dataset/downloads/manual
   """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(custom_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'objects': tfds.features.Sequence({
                                  'bbox': tfds.features.BBoxFeature(),
                                  'label': tfds.features.ClassLabel(names=['Choi Woo-shik',
                                                                           'Kim Da-mi',
                                                                           'Kim Seong-cheol',
                                                                           'Kim Tae-ri',
                                                                           'Nam Joo-hyuk',
                                                                           'Yoo Jae-suk']),
                                  'is_difficult': tfds.features.Tensor(shape=(), dtype=tf.bool)
            })
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'objects'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(custom_dataset): Downloads the data and defines the splits
    archive_path = dl_manager.manual_dir / 'data.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(custom_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(img_path=extracted_path / 'train_images',
                                         xml_path=extracted_path / 'train_xmls'),
        'test': self._generate_examples(img_path=extracted_path / 'test_images',
                                        xml_path=extracted_path / 'test_xmls'),
    }

  def _generate_examples(self, img_path, xml_path):
    """Yields examples."""
    # TODO(custom_dataset): Yields (key, example) tuples from the dataset
    for i, (img, xml) in enumerate(zip(img_path.glob('*.jpg'), xml_path.glob('*.xml'))):
      yield i,{
        'image': img,
        'objects': self._get_objects(xml)
      }
  
  def _get_image(self, img):
    image=np.array(Image.open(img))
    return image
  def _get_objects(self, xml):
    data=dict()
    f=open(xml)
    xml_file=xmltodict.parse(f.read())
    bbox=[]
    label=[]
    is_difficult=[]
    height, width = xml_file['annotation']['size']['height'], xml_file['annotation']['size']['width']
    for obj in xml_file['annotation']['object']:
      if type(obj)==type(dict()):
        label.append(obj['name'])
        is_difficult.append(bool(int(obj['difficult'])))
        x1=obj['bndbox']['xmin']
        y1=obj['bndbox']['ymin']
        x2=obj['bndbox']['xmax']
        y2=obj['bndbox']['ymax']
        y1, y2 = float(y1)/float(height), float(y2)/float(height)
        x1, x2 = float(x1)/float(width), float(x2)/float(width)
        bbox.append(tfds.features.BBox(ymin=y1, xmin=x1, ymax=y2, xmax=x2))
      else:
        if obj=='name':
          label.append(xml_file['annotation']['object'][obj])
        elif obj=='difficult':
          is_difficult.append(bool(int(xml_file['annotation']['object'][obj])))
        elif obj=='bndbox':
          x1 = xml_file['annotation']['object'][obj]['xmin']
          y1 = xml_file['annotation']['object'][obj]['ymin']
          x2 = xml_file['annotation']['object'][obj]['xmax']
          y2 = xml_file['annotation']['object'][obj]['ymax']
          y1, y2 = float(y1)/float(height), float(y2)/float(height)
          x1, x2 = float(x1)/float(width), float(x2)/float(width)
          bbox.append(tfds.features.BBox(ymin=y1, xmin=x1, ymax=y2, xmax=x2))
    f.close()
    data['bbox']=bbox
    data['label']=label
    data['is_difficult']=is_difficult
    return data