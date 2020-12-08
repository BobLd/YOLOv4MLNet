using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace YOLOv4MLNet.DataStructures
{
    public class YoloV4Prediction
    {
        // https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/anchors/yolov4_anchors.txt
        static readonly float[][][] ANCHORS = new float[][][]
        {
            new float[][] { new float[] { 10, 13 }, new float[] { 16, 30 }, new float[] { 33, 23 } },
            new float[][] { new float[] { 30, 61 }, new float[] { 62, 45 }, new float[] { 59, 119 } },
            new float[][] { new float[] { 116, 90 }, new float[] { 156, 198 }, new float[] { 373, 326 } }
        };

        // https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/core/config.py#L18
        static readonly float[] STRIDES = new float[] { 8, 16, 32 }; // 640/80, 640/40, 640/20

        // https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/core/config.py#L20
        static readonly float[] XYSCALE = new float[] { 1f, 1f, 1f };

        static readonly int[] shapes = new int[] { 80, 40, 20 };

        const int anchorsCount = 3;

        /// <summary>
        /// Identity
        /// </summary>
        [VectorType(1, 3, 80, 80, 85)]
        [ColumnName("output")]
        public float[] Output { get; set; }

        /// <summary>
        /// Identity1
        /// </summary>
        [VectorType(1, 3, 40, 40, 85)]
        [ColumnName("1313")]
        public float[] Output1313 { get; set; }

        /// <summary>
        /// Identity2
        /// </summary>
        [VectorType(1, 3, 20, 20, 85)]
        [ColumnName("1333")]
        public float[] Output1333 { get; set; }

        [ColumnName("width")]
        public float ImageWidth { get; set; }

        [ColumnName("height")]
        public float ImageHeight { get; set; }

        public IReadOnlyList<YoloV4Result> GetResults(string[] categories, float scoreThres = 0.5f, float iouThres = 0.5f)
        {
            List<float[]> postProcesssedResults = new List<float[]>();
            int classesCount = categories.Length;
            var results = new[] { Output, Output1313, Output1333 };

            for (int i = 0; i < results.Length; i++)
            {
                var pred = results[i];
                var outputSize = shapes[i];
                for (int boxY = 0; boxY < outputSize; boxY++)
                {
                    for (int boxX = 0; boxX < outputSize; boxX++)
                    {
                        for (int a = 0; a < anchorsCount; a++)
                        {
                            var offset = (boxY * outputSize * (classesCount + 5) * anchorsCount) + (boxX * (classesCount + 5) * anchorsCount) + a * (classesCount + 5);
                            var predBbox = pred.Skip(offset).Take(classesCount + 5).Select(x => Sigmoid(x)).ToArray(); // y = x[i].sigmoid()

                            // more info at 
                            // https://github.com/ultralytics/yolov5/issues/343#issuecomment-658021043
                            // https://github.com/ultralytics/yolov5/blob/a1c8406af3eac3e20d4dd5d327fd6cbd4fbb9752/models/yolo.py#L29-L36

                            // postprocess_bbbox()
                            var predXywh = predBbox.Take(4).ToArray();
                            var predConf = predBbox[4];
                            var predProb = predBbox.Skip(5).ToArray();

                            var rawDx = predXywh[0];
                            var rawDy = predXywh[1];
                            var rawDw = predXywh[2];
                            var rawDh = predXywh[3];

                            float predX = ((rawDx * 2f) - 0.5f + boxX) * STRIDES[i];
                            float predY = ((rawDy * 2f) - 0.5f + boxY) * STRIDES[i];
                            float predW = (float)Math.Pow(rawDw * 2, 2) * ANCHORS[i][a][0];
                            float predH = (float)Math.Pow(rawDh * 2, 2) * ANCHORS[i][a][1];

                            // postprocess_boxes
                            // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
                            var box = Xywh2xyxy(new float[] { predX, predY, predW, predH });
                            float predX1 = box[0]; //predX - predW * 0.5f;
                            float predY1 = box[1]; //predY - predH * 0.5f;
                            float predX2 = box[2]; //predX + predW * 0.5f;
                            float predY2 = box[3]; //predY + predH * 0.5f;

                            // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
                            float org_h = ImageHeight;
                            float org_w = ImageWidth;

                            float inputSize = 640f;
                            float resizeRatio = Math.Min(inputSize / org_w, inputSize / org_h);
                            float dw = (inputSize - resizeRatio * org_w) / 2f;
                            float dh = (inputSize - resizeRatio * org_h) / 2f;

                            var orgX1 = 1f * (predX1 - dw) / resizeRatio; // left
                            var orgX2 = 1f * (predX2 - dw) / resizeRatio; // right
                            var orgY1 = 1f * (predY1 - dh) / resizeRatio; // top
                            var orgY2 = 1f * (predY2 - dh) / resizeRatio; // bottom

                            // (3) clip some boxes that are out of range
                            orgX1 = Math.Max(orgX1, 0);
                            orgY1 = Math.Max(orgY1, 0);
                            orgX2 = Math.Min(orgX2, org_w - 1);
                            orgY2 = Math.Min(orgY2, org_h - 1);
                            if (orgX1 > orgX2 || orgY1 > orgY2) continue; // invalid_mask

                            // (4) discard some invalid boxes
                            // TODO

                            // (5) discard some boxes with low scores
                            var scores = predProb.Select(p => p * predConf).ToList();

                            float scoreMaxCat = scores.Max();
                            if (scoreMaxCat > scoreThres)
                            {
                                postProcesssedResults.Add(new float[] { orgX1, orgY1, orgX2, orgY2, scoreMaxCat, scores.IndexOf(scoreMaxCat) });
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
                        //postProcesssedResults[i] = null; // deactivated for debugging
                    }
                }
                f++;
            }

            return resultsNms;
        }

        /// <summary>
        /// expit = https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
        /// </summary>
        private static float Sigmoid(float x)
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

        /// <summary>
        /// Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        /// <para>Box (center x, center y, width, height) to (x1, y1, x2, y2)</para>
        /// </summary>
        public static float[] Xywh2xyxy(float[] bbox)
        {
            //https://github.com/BobLd/YOLOv3MLNet/blob/48d691175249c2dbf1fdfb9790c9aec56f9a028f/YOLOv3MLNet/DataStructures/YoloV3Prediction.cs#L159-L167
            var bboxAdj = new float[4];
            bboxAdj[0] = bbox[0] - bbox[2] / 2f;
            bboxAdj[1] = bbox[1] - bbox[3] / 2f;
            bboxAdj[2] = bbox[0] + bbox[2] / 2f;
            bboxAdj[3] = bbox[1] + bbox[3] / 2f;
            return bboxAdj;
        }
    }
}
