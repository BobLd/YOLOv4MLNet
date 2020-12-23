using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace YOLOv4MLNet.DataStructures
{
    public class YoloV4Prediction
    {
        /// <summary>
        /// Identity
        /// </summary>
        [VectorType(1, 25200, 85)]
        [ColumnName("output")]
        public float[] Output { get; set; }

        [ColumnName("width")]
        public float ImageWidth { get; set; }

        [ColumnName("height")]
        public float ImageHeight { get; set; }

        public IReadOnlyList<YoloV5Result> GetResults(string[] categories, float scoreThres = 0.5f, float iouThres = 0.5f)
        {

            // Probabilities + Characteristics
            int characteristics = categories.Length + 5;

            // Needed info
            float modelWidth = 640.0F;
            float modelHeight = 640.0F;
            float xGain = modelWidth / ImageWidth;
            float yGain = modelHeight / ImageHeight;
            float[] results = Output;

            List<float[]> postProcessedResults = new List<float[]>();

            // For every cell of the image, format for NMS
            for (int i = 0; i < 25200; i++)
            {
                // Get offset in float array
                int offset = characteristics * i;

                // Get a prediction cell
                var predCell = results.Skip(offset).Take(characteristics).ToList();

                // Filter some boxes
                var objConf = predCell[4];
                if (objConf <= scoreThres) continue;

                // Get corners in original shape
                var x1 = (predCell[0] - predCell[2] / 2) / xGain; //top left x
                var y1 = (predCell[1] - predCell[3] / 2) / yGain; //top left y
                var x2 = (predCell[0] + predCell[2] / 2) / xGain; //bottom right x
                var y2 = (predCell[1] + predCell[3] / 2) / yGain; //bottom right y

                // Get real class scores
                var classProbs = predCell.Skip(5).Take(categories.Length).ToList();
                var scores = classProbs.Select(p => p * objConf).ToList();

                // Get best class and index
                float maxConf = scores.Max();
                float maxClass = scores.ToList().IndexOf(maxConf);

                postProcessedResults.Add(new[] { x1, y1, x2, y2, maxConf, maxClass });
            }

            var resultsNMS = ApplyNMS(postProcessedResults, categories, iouThres);

            return resultsNMS;
        }

        private List<YoloV5Result> ApplyNMS(List<float[]> postProcessedResults, string[] categories, float iouThres = 0.5f)
        {
            postProcessedResults = postProcessedResults.OrderByDescending(x => x[4]).ToList(); // sort by confidence
            List<YoloV5Result> resultsNms = new List<YoloV5Result>();

            int f = 0;
            while (f < postProcessedResults.Count)
            {
                var res = postProcessedResults[f];
                if (res == null)
                {
                    f++;
                    continue;
                }

                var conf = res[4];
                string label = categories[(int)res[5]];

                resultsNms.Add(new YoloV5Result(res.Take(4).ToArray(), label, conf));
                postProcessedResults[f] = null;

                var iou = postProcessedResults.Select(bbox => bbox == null ? float.NaN : BoxIoU(res, bbox)).ToList();
                for (int i = 0; i < iou.Count; i++)
                {
                    if (float.IsNaN(iou[i])) continue;
                    if (iou[i] > iouThres)
                    {
                        postProcessedResults[i] = null;
                    }
                }
                f++;
            }

            return resultsNms;
        }

        /// <summary>
        /// Return intersection-over-union (Jaccard index) of boxes.
        /// <para>Both sets of boxes are expected to be in (x1, y1, x2, y2) format.</para>
        /// </summary>
        private static float BoxIoU(float[] boxes1, float[] boxes2)
        {
            static float box_area(float[] box)
            {
                return (box[2] - box[0]) * (box[3] - box[1]);
            }

            var area1 = box_area(boxes1);
            var area2 = box_area(boxes2);

            Debug.Assert(area1 >= 0);
            Debug.Assert(area2 >= 0);

            var dx = Math.Max(0, Math.Min(boxes1[2], boxes2[2]) - Math.Max(boxes1[0], boxes2[0]));
            var dy = Math.Max(0, Math.Min(boxes1[3], boxes2[3]) - Math.Max(boxes1[1], boxes2[1]));
            var inter = dx * dy;

            return inter / (area1 + area2 - inter);
        }
    }
}
