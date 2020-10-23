using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace YOLOv4MLNet.DataStructures
{
    public class YoloV4BitmapData
    {
        [ColumnName("bitmap")]
        [ImageType(416, 416)]
        public Bitmap Image { get; set; }

        [ColumnName("width")]
        public float ImageWidth => Image.Width;

        [ColumnName("height")]
        public float ImageHeight => Image.Height;
    }
}
