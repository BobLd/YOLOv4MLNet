using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace YOLOv4MLNet.DataStructures
{
    public class YoloV4Prediction
    {
        //https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/anchors/yolov4_anchors.txt
        static readonly float[][][] ANCHORS = new float[][][]
        {
            new float[][] { new float[] { 12, 16 }, new float[] { 19, 36 }, new float[] { 40, 28 } },
            new float[][] { new float[] { 36, 75 }, new float[] { 76, 55 }, new float[] { 72, 146 } },
            new float[][] { new float[] { 142, 110 }, new float[] { 192, 243 }, new float[] { 459, 401 } }
        };

        static readonly float[] STRIDES = new float[] { 8, 16, 32 };

        static readonly int[] shapes = new int[] { 52, 26, 13 };

        static readonly float[] XYSCALE = new float[] { 1.2f, 1.1f, 1.05f };

        const int anchorsCount = 3;

        /// <summary>
        /// Identity
        /// </summary>
        [VectorType(1, 52, 52, 3, 85)]
        [ColumnName("Identity:0")]
        public float[] Identity { get; set; }

        /// <summary>
        /// Identity1
        /// </summary>
        [VectorType(1, 26, 26, 3, 85)]
        [ColumnName("Identity_1:0")]
        public float[] Identity1 { get; set; }

        /// <summary>
        /// Identity2
        /// </summary>
        [VectorType(1, 13, 13, 3, 85)]
        [ColumnName("Identity_2:0")]
        public float[] Identity2 { get; set; }

        [ColumnName("width")]
        public float ImageWidth { get; set; }

        [ColumnName("height")]
        public float ImageHeight { get; set; }

        public IReadOnlyList<YoloV4Result> GetResults(string[] categories, float scoreThres = 0.5f, float iouThres = 0.5f)
        {
            List<float[]> postProcesssedResults = new List<float[]>();
            int classesCount = categories.Length;
            var results = new[] { Identity, Identity1, Identity2 };
            for (int i = 0; i < results.Length; i++)
            {
                var pred = results[i];
                var output_size = shapes[i];

                for (int boxY = 0; boxY < output_size; boxY++)
                {
                    for (int boxX = 0; boxX < output_size; boxX++)
                    {
                        for (int a = 0; a < anchorsCount; a++) // anchors
                        {
                            var offset = (boxY * output_size * (classesCount + 5) * anchorsCount) + (boxX * (classesCount + 5) * anchorsCount) + a * (classesCount + 5);
                            var pred_bbox = pred.Skip(offset).Take(classesCount + 5).ToArray();

                            // postprocess_bbbox()
                            var pred_xywh = pred_bbox.Take(4).ToArray();
                            var pred_conf = pred_bbox[4];
                            var pred_prob = pred_bbox.Skip(5).ToArray();

                            var raw_dx = pred_xywh[0];
                            var raw_dy = pred_xywh[1];
                            var raw_dw = pred_xywh[2];
                            var raw_dh = pred_xywh[3];

                            float pred_x = ((expit(raw_dx) * XYSCALE[i]) - 0.5f * (XYSCALE[i] - 1) + boxX) * STRIDES[i];
                            float pred_y = ((expit(raw_dy) * XYSCALE[i]) - 0.5f * (XYSCALE[i] - 1) + boxY) * STRIDES[i];
                            float pred_w = (float)Math.Exp(raw_dw) * ANCHORS[i][a][0];
                            float pred_h = (float)Math.Exp(raw_dh) * ANCHORS[i][a][1];

                            // postprocess_boxes
                            // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
                            float pred_x1 = pred_x - pred_w * 0.5f;
                            float pred_y1 = pred_y - pred_h * 0.5f;
                            float pred_x2 = pred_x + pred_w * 0.5f;
                            float pred_y2 = pred_y + pred_h * 0.5f;

                            // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
                            float org_h = ImageHeight;
                            float org_w = ImageWidth;

                            float input_size = 416f;
                            float resize_ratio = Math.Min(input_size / org_w, input_size / org_h);
                            float dw = (input_size - resize_ratio * org_w) / 2f;
                            float dh = (input_size - resize_ratio * org_h) / 2f;

                            var org_x1 = 1f * (pred_x1 - dw) / resize_ratio;
                            var org_x2 = 1f * (pred_x2 - dw) / resize_ratio;
                            var org_y1 = 1f * (pred_y1 - dh) / resize_ratio;
                            var org_y2 = 1f * (pred_y2 - dh) / resize_ratio;

                            // (3) clip some boxes that are out of range
                            // TODO

                            // (4) discard some invalid boxes
                            // TODO

                            // (5) discard some boxes with low scores
                            var scores = pred_prob.Select(p => p * pred_conf).ToList();

                            float score_max_cat = scores.Max();
                            if (score_max_cat > scoreThres)
                            {
                                postProcesssedResults.Add(new float[] { org_x1, org_y1, org_x2, org_y2, score_max_cat, scores.IndexOf(score_max_cat) });
                            }
                        }
                    }
                }
            }

            // Non-maximum Suppression
            postProcesssedResults = postProcesssedResults.OrderByDescending(x => x[4]).ToList(); // sort by confidence
            List<YoloV4Result> resultsNms = new List<YoloV4Result>();

            int f = 0;
            while (f < postProcesssedResults.Count)
            {
                var res = postProcesssedResults[f];
                if (res == null)
                {
                    f++;
                    continue;
                }

                var conf = res[4];
                string label = categories[(int)res[5]];

                resultsNms.Add(new YoloV4Result(res.Take(4).ToArray(), label, conf));
                postProcesssedResults[f] = null;

                var iou = postProcesssedResults.Select(bbox => bbox == null ? float.NaN : BoxIoU(res, bbox)).ToList();
                for (int i = 0; i < iou.Count; i++)
                {
                    if (float.IsNaN(iou[i])) continue;
                    if (iou[i] > iouThres)
                    {
                        postProcesssedResults[i] = null;
                    }
                }
                f++;
            }

            return resultsNms;
        }

        /// <summary>
        /// https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
        /// </summary>
        private static float expit(float x)
        {
            return 1f / (1f + (float)Math.Exp(-x));
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
