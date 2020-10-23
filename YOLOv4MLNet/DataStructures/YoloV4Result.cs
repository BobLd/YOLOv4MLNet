namespace YOLOv4MLNet.DataStructures
{
    public class YoloV4Result
    {
        /// <summary>
        /// x1, y1, x2, y2 in page coordinates.
        /// </summary>
        public float[] BBox { get; }

        /// <summary>
        /// The Bbox category.
        /// </summary>
        public string Label { get; }

        /// <summary>
        /// Category's confidence level.
        /// </summary>
        public float Confidence { get; }

        public YoloV4Result(float[] bbox, string label, float confidence)
        {
            BBox = bbox;
            Label = label;
            Confidence = confidence;
        }
    }
}
