using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace NelderMeadMethod
{
    public class Simplex
    {
        public double[][] Vertexes { get; private set; }
        public double[] Values { get; private set; }
        public int Dimension { get; }
        public int VertexCount { get; }

        public Simplex(double[] startPoint, double step)
        {
            if (startPoint == null)
                throw new ArgumentNullException(nameof(startPoint));

            if (startPoint.Length == 0)
                throw new ArgumentException("Начальная точка не должна быть пустой.", nameof(startPoint));

            if (step <= 0)
                throw new ArgumentOutOfRangeException(nameof(step), "Шаг должен быть положительным.");

            Dimension = startPoint.Length;
            VertexCount = Dimension + 1;

            Vertexes = new double[VertexCount][];
            Values = new double[VertexCount];

            Vertexes[0] = new double[Dimension];
            Array.Copy(startPoint, Vertexes[0], Dimension);

            for (int i = 1; i < VertexCount; i++)
            {
                Vertexes[i] = new double[Dimension];
                Array.Copy(startPoint, Vertexes[i], Dimension);
                Vertexes[i][i - 1] += step;
            }
        }

        public void UpdateValues(Func<double[], double> objectiveFunction)
        {
            if (objectiveFunction == null)
                throw new ArgumentNullException(nameof(objectiveFunction));

            for (int i = 0; i < VertexCount; i++)
            {
                Values[i] = objectiveFunction(Vertexes[i]);
            }
        }

        public double GetValue(int vertexIndex) => Values[vertexIndex];

        public double[] GetWorstVertex()
        {
            double[] result = new double[Dimension];
            Array.Copy(Vertexes[VertexCount - 1], result, Dimension);
            return result;
        }

        public double[] GetBestVertex()
        {
            double[] result = new double[Dimension];
            Array.Copy(Vertexes[0], result, Dimension);
            return result;
        }

        public double GetSecondWorstValue() => Values[VertexCount - 2];

        public void ReplaceWorstVertex(double[] newVertex, double newValue)
        {
            if (newVertex == null)
                throw new ArgumentNullException(nameof(newVertex));

            if (newVertex.Length != Dimension)
                throw new ArgumentException("Размерность вершины не совпадает с размерностью симплекса.", nameof(newVertex));

            Array.Copy(newVertex, Vertexes[VertexCount - 1], Dimension);
            Values[VertexCount - 1] = newValue;
        }

        public double[] CalculateCentroid()
        {
            double[] centroid = new double[Dimension];

            for (int i = 0; i < VertexCount - 1; i++)
            {
                for (int j = 0; j < Dimension; j++)
                {
                    centroid[j] += Vertexes[i][j];
                }
            }

            for (int j = 0; j < Dimension; j++)
            {
                centroid[j] /= VertexCount - 1;
            }

            return centroid;
        }

        public bool IsConverged(double accuracy)
        {
            if (accuracy <= 0)
                throw new ArgumentOutOfRangeException(nameof(accuracy), "Точность должна быть положительной.");

            return Math.Abs(Values[VertexCount - 1] - Values[0]) < accuracy;
        }

        public void SortVertexes()
        {
            var indexedValues = Values
                .Select((value, index) => new { Value = value, Index = index })
                .OrderBy(x => x.Value)
                .ToList();

            double[][] sortedVertexes = new double[VertexCount][];
            double[] sortedValues = new double[VertexCount];

            for (int i = 0; i < VertexCount; i++)
            {
                int originalIndex = indexedValues[i].Index;
                sortedVertexes[i] = Vertexes[originalIndex];
                sortedValues[i] = Values[originalIndex];
            }

            Vertexes = sortedVertexes;
            Values = sortedValues;
        }

        public void Shrink(Func<double[], double> objectiveFunction, double sigma = 0.5)
        {
            if (objectiveFunction == null)
                throw new ArgumentNullException(nameof(objectiveFunction));

            if (sigma <= 0 || sigma >= 1)
                throw new ArgumentOutOfRangeException(nameof(sigma), "Коэффициент shrink должен быть в диапазоне (0, 1).");

            double[] best = GetBestVertex();

            for (int i = 1; i < VertexCount; i++)
            {
                for (int j = 0; j < Dimension; j++)
                {
                    Vertexes[i][j] = best[j] + sigma * (Vertexes[i][j] - best[j]);
                }

                Values[i] = objectiveFunction(Vertexes[i]);
            }

            Values[0] = objectiveFunction(Vertexes[0]);
            SortVertexes();
        }
    }

    public class NelderMeadOptimizer
    {
        public double ReflectionCoeff { get; set; } = 1.0;
        public double ExpansionCoeff { get; set; } = 2.0;
        public double ContractionCoeff { get; set; } = 0.5;
        public double ShrinkCoeff { get; set; } = 0.5;
        public double Step { get; set; } = 0.5;
        public double Accuracy { get; set; } = 1e-6;

        public List<double[][]> History { get; } = new List<double[][]>();

        private readonly Func<double[], double> objectiveFunction;

        public NelderMeadOptimizer(Func<double[], double> objectiveFunction)
        {
            this.objectiveFunction = objectiveFunction ?? throw new ArgumentNullException(nameof(objectiveFunction));
        }

        private void SaveSimplex(Simplex simplex)
        {
            double[][] copy = new double[simplex.VertexCount][];

            for (int i = 0; i < simplex.VertexCount; i++)
            {
                copy[i] = new double[simplex.Dimension];
                Array.Copy(simplex.Vertexes[i], copy[i], simplex.Dimension);
            }

            History.Add(copy);
        }

        public double[] Optimize(double[] startPoint, int maxIterations)
        {
            if (startPoint == null)
                throw new ArgumentNullException(nameof(startPoint));

            if (startPoint.Length == 0)
                throw new ArgumentException("Начальная точка не должна быть пустой.", nameof(startPoint));

            if (maxIterations <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxIterations), "Число итераций должно быть положительным.");

            if (Step <= 0)
                throw new ArgumentOutOfRangeException(nameof(Step), "Шаг должен быть положительным.");

            if (Accuracy <= 0)
                throw new ArgumentOutOfRangeException(nameof(Accuracy), "Точность должна быть положительной.");

            History.Clear();

            Simplex simplex = new Simplex(startPoint, Step);
            simplex.UpdateValues(objectiveFunction);
            simplex.SortVertexes();
            SaveSimplex(simplex);

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                if (simplex.IsConverged(Accuracy))
                    break;

                double[] centroid = simplex.CalculateCentroid();
                double[] worst = simplex.GetWorstVertex();

                double bestValue = simplex.GetValue(0);
                double secondWorstValue = simplex.GetSecondWorstValue();
                double worstValue = simplex.GetValue(simplex.VertexCount - 1);

                double[] reflected = new double[simplex.Dimension];

                for (int j = 0; j < simplex.Dimension; j++)
                {
                    reflected[j] = centroid[j] + ReflectionCoeff * (centroid[j] - worst[j]);
                }

                double reflectedValue = objectiveFunction(reflected);

                if (reflectedValue < bestValue)
                {
                    double[] expanded = new double[simplex.Dimension];

                    for (int j = 0; j < simplex.Dimension; j++)
                    {
                        expanded[j] = centroid[j] + ExpansionCoeff * (reflected[j] - centroid[j]);
                    }

                    double expandedValue = objectiveFunction(expanded);

                    if (expandedValue < reflectedValue)
                        simplex.ReplaceWorstVertex(expanded, expandedValue);
                    else
                        simplex.ReplaceWorstVertex(reflected, reflectedValue);

                    simplex.SortVertexes();
                    SaveSimplex(simplex);
                    continue;
                }

                if (reflectedValue < secondWorstValue)
                {
                    simplex.ReplaceWorstVertex(reflected, reflectedValue);
                    simplex.SortVertexes();
                    SaveSimplex(simplex);
                    continue;
                }

                double[] contracted = new double[simplex.Dimension];

                if (reflectedValue < worstValue)
                {
                    for (int j = 0; j < simplex.Dimension; j++)
                    {
                        contracted[j] = centroid[j] + ContractionCoeff * (reflected[j] - centroid[j]);
                    }
                }
                else
                {
                    for (int j = 0; j < simplex.Dimension; j++)
                    {
                        contracted[j] = centroid[j] - ContractionCoeff * (centroid[j] - worst[j]);
                    }
                }

                double contractedValue = objectiveFunction(contracted);

                if (contractedValue < worstValue)
                {
                    simplex.ReplaceWorstVertex(contracted, contractedValue);
                    simplex.SortVertexes();
                    SaveSimplex(simplex);
                    continue;
                }

                simplex.Shrink(objectiveFunction, ShrinkCoeff);
                SaveSimplex(simplex);
            }

            return simplex.GetBestVertex();
        }
    }

    public static class SimplexVisualizer
    {
        private const int PngWidth = 900;
        private const int PngHeight = 700;

        public static void SaveEvolutionPng(List<double[][]> history, string path)
        {
            byte[] image = CreateWhiteImage(PngWidth, PngHeight);

            int cols = 3;
            int rows = 3;
            int cellWidth = PngWidth / cols;
            int cellHeight = PngHeight / rows;

            int frames = Math.Min(9, history.Count);

            for (int i = 0; i < frames; i++)
            {
                int index = history.Count == 1 ? 0 : i * (history.Count - 1) / (frames - 1);

                int col = i % cols;
                int row = i / cols;

                DrawSimplexFrame(
                    image,
                    PngWidth,
                    PngHeight,
                    history[index],
                    col * cellWidth,
                    row * cellHeight,
                    cellWidth,
                    cellHeight);
            }

            SavePng(image, PngWidth, PngHeight, path);
        }

        public static void SaveAnimationGif(List<double[][]> history, string path)
        {
            int width = 500;
            int height = 500;

            using FileStream stream = new FileStream(path, FileMode.Create);
            using BinaryWriter writer = new BinaryWriter(stream);

            writer.Write(Encoding.ASCII.GetBytes("GIF89a"));
            WriteShort(writer, width);
            WriteShort(writer, height);

            writer.Write((byte)0xF7);
            writer.Write((byte)0);
            writer.Write((byte)0);

            WritePalette(writer);

            writer.Write((byte)0x21);
            writer.Write((byte)0xFF);
            writer.Write((byte)11);
            writer.Write(Encoding.ASCII.GetBytes("NETSCAPE2.0"));
            writer.Write((byte)3);
            writer.Write((byte)1);
            WriteShort(writer, 0);
            writer.Write((byte)0);

            int step = Math.Max(1, history.Count / 20);

            for (int i = 0; i < history.Count; i += step)
            {
                byte[] frame = CreateGifFrame(history[i], width, height);
                WriteGifFrame(writer, frame, width, height, 12);
            }

            writer.Write((byte)0x3B);
        }

        private static void DrawSimplexFrame(
            byte[] image,
            int width,
            int height,
            double[][] simplex,
            int offsetX,
            int offsetY,
            int cellWidth,
            int cellHeight)
        {
            int left = offsetX + 25;
            int right = offsetX + cellWidth - 25;
            int top = offsetY + 25;
            int bottom = offsetY + cellHeight - 25;

            DrawLine(image, width, height, left, bottom, right, bottom, 0, 0, 0);
            DrawLine(image, width, height, left, top, left, bottom, 0, 0, 0);

            List<(int X, int Y)> points = new List<(int X, int Y)>();

            foreach (double[] vertex in simplex)
            {
                int x = Map(vertex[0], -0.5, 2.0, left, right);
                int y = Map(vertex[1], -0.5, 2.0, bottom, top);
                points.Add((x, y));
            }

            for (int i = 0; i < points.Count; i++)
            {
                int next = (i + 1) % points.Count;
                DrawLine(image, width, height, points[i].X, points[i].Y, points[next].X, points[next].Y, 30, 80, 220);
            }

            foreach ((int x, int y) in points)
            {
                DrawCircle(image, width, height, x, y, 4, 220, 0, 0);
            }
        }

        private static byte[] CreateGifFrame(double[][] simplex, int width, int height)
        {
            byte[] image = new byte[width * height];

            int left = 50;
            int right = width - 50;
            int top = 50;
            int bottom = height - 50;

            DrawIndexedLine(image, width, height, left, bottom, right, bottom, 1);
            DrawIndexedLine(image, width, height, left, top, left, bottom, 1);

            List<(int X, int Y)> points = new List<(int X, int Y)>();

            foreach (double[] vertex in simplex)
            {
                int x = Map(vertex[0], -0.5, 2.0, left, right);
                int y = Map(vertex[1], -0.5, 2.0, bottom, top);
                points.Add((x, y));
            }

            for (int i = 0; i < points.Count; i++)
            {
                int next = (i + 1) % points.Count;
                DrawIndexedLine(image, width, height, points[i].X, points[i].Y, points[next].X, points[next].Y, 2);
            }

            foreach ((int x, int y) in points)
            {
                DrawIndexedCircle(image, width, height, x, y, 5, 3);
            }

            return image;
        }

        private static int Map(double value, double minValue, double maxValue, int minPixel, int maxPixel)
        {
            double t = (value - minValue) / (maxValue - minValue);
            t = Math.Max(0.0, Math.Min(1.0, t));

            return minPixel + (int)(t * (maxPixel - minPixel));
        }

        private static byte[] CreateWhiteImage(int width, int height)
        {
            byte[] image = new byte[width * height * 3];

            for (int i = 0; i < image.Length; i++)
            {
                image[i] = 255;
            }

            return image;
        }

        private static void DrawCircle(byte[] image, int width, int height, int cx, int cy, int r, byte red, byte green, byte blue)
        {
            for (int y = cy - r; y <= cy + r; y++)
            {
                for (int x = cx - r; x <= cx + r; x++)
                {
                    int dx = x - cx;
                    int dy = y - cy;

                    if (dx * dx + dy * dy <= r * r)
                    {
                        SetPixel(image, width, height, x, y, red, green, blue);
                    }
                }
            }
        }

        private static void DrawLine(byte[] image, int width, int height, int x0, int y0, int x1, int y1, byte red, byte green, byte blue)
        {
            int dx = Math.Abs(x1 - x0);
            int dy = -Math.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int error = dx + dy;

            while (true)
            {
                SetPixel(image, width, height, x0, y0, red, green, blue);

                if (x0 == x1 && y0 == y1)
                    break;

                int e2 = 2 * error;

                if (e2 >= dy)
                {
                    error += dy;
                    x0 += sx;
                }

                if (e2 <= dx)
                {
                    error += dx;
                    y0 += sy;
                }
            }
        }

        private static void SetPixel(byte[] image, int width, int height, int x, int y, byte red, byte green, byte blue)
        {
            if (x < 0 || x >= width || y < 0 || y >= height)
                return;

            int index = (y * width + x) * 3;
            image[index] = red;
            image[index + 1] = green;
            image[index + 2] = blue;
        }

        private static void DrawIndexedCircle(byte[] image, int width, int height, int cx, int cy, int r, byte color)
        {
            for (int y = cy - r; y <= cy + r; y++)
            {
                for (int x = cx - r; x <= cx + r; x++)
                {
                    int dx = x - cx;
                    int dy = y - cy;

                    if (dx * dx + dy * dy <= r * r)
                    {
                        SetIndexedPixel(image, width, height, x, y, color);
                    }
                }
            }
        }

        private static void DrawIndexedLine(byte[] image, int width, int height, int x0, int y0, int x1, int y1, byte color)
        {
            int dx = Math.Abs(x1 - x0);
            int dy = -Math.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int error = dx + dy;

            while (true)
            {
                SetIndexedPixel(image, width, height, x0, y0, color);

                if (x0 == x1 && y0 == y1)
                    break;

                int e2 = 2 * error;

                if (e2 >= dy)
                {
                    error += dy;
                    x0 += sx;
                }

                if (e2 <= dx)
                {
                    error += dx;
                    y0 += sy;
                }
            }
        }

        private static void SetIndexedPixel(byte[] image, int width, int height, int x, int y, byte color)
        {
            if (x < 0 || x >= width || y < 0 || y >= height)
                return;

            image[y * width + x] = color;
        }

        private static void SavePng(byte[] rgb, int width, int height, string path)
        {
            using FileStream file = new FileStream(path, FileMode.Create);
            using BinaryWriter writer = new BinaryWriter(file);

            writer.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });

            byte[] ihdr = new byte[13];
            WriteInt(ihdr, 0, width);
            WriteInt(ihdr, 4, height);
            ihdr[8] = 8;
            ihdr[9] = 2;
            ihdr[10] = 0;
            ihdr[11] = 0;
            ihdr[12] = 0;

            WritePngChunk(writer, "IHDR", ihdr);

            using MemoryStream rawStream = new MemoryStream();

            for (int y = 0; y < height; y++)
            {
                rawStream.WriteByte(0);
                rawStream.Write(rgb, y * width * 3, width * 3);
            }

            using MemoryStream compressedStream = new MemoryStream();

            using (ZLibStream zlib = new ZLibStream(compressedStream, CompressionLevel.Optimal, true))
            {
                byte[] raw = rawStream.ToArray();
                zlib.Write(raw, 0, raw.Length);
            }

            WritePngChunk(writer, "IDAT", compressedStream.ToArray());
            WritePngChunk(writer, "IEND", Array.Empty<byte>());
        }

        private static void WritePngChunk(BinaryWriter writer, string type, byte[] data)
        {
            WriteInt(writer, data.Length);

            byte[] typeBytes = Encoding.ASCII.GetBytes(type);
            writer.Write(typeBytes);
            writer.Write(data);

            byte[] crcData = new byte[typeBytes.Length + data.Length];
            Array.Copy(typeBytes, 0, crcData, 0, typeBytes.Length);
            Array.Copy(data, 0, crcData, typeBytes.Length, data.Length);

            WriteInt(writer, unchecked((int)Crc32(crcData)));
        }

        private static void WriteInt(BinaryWriter writer, int value)
        {
            writer.Write(new[]
            {
                (byte)((value >> 24) & 255),
                (byte)((value >> 16) & 255),
                (byte)((value >> 8) & 255),
                (byte)(value & 255)
            });
        }

        private static void WriteInt(byte[] buffer, int offset, int value)
        {
            buffer[offset] = (byte)((value >> 24) & 255);
            buffer[offset + 1] = (byte)((value >> 16) & 255);
            buffer[offset + 2] = (byte)((value >> 8) & 255);
            buffer[offset + 3] = (byte)(value & 255);
        }

        private static uint Crc32(byte[] data)
        {
            uint crc = 0xffffffff;

            foreach (byte b in data)
            {
                crc ^= b;

                for (int i = 0; i < 8; i++)
                {
                    if ((crc & 1) == 1)
                    {
                        crc = (crc >> 1) ^ 0xedb88320;
                    }
                    else
                    {
                        crc >>= 1;
                    }
                }
            }

            return crc ^ 0xffffffff;
        }

        private static void WriteShort(BinaryWriter writer, int value)
        {
            writer.Write((byte)(value & 255));
            writer.Write((byte)((value >> 8) & 255));
        }

        private static void WritePalette(BinaryWriter writer)
        {
            byte[][] colors =
            {
                new byte[] { 255, 255, 255 },
                new byte[] { 0, 0, 0 },
                new byte[] { 30, 80, 220 },
                new byte[] { 220, 0, 0 }
            };

            for (int i = 0; i < 256; i++)
            {
                if (i < colors.Length)
                    writer.Write(colors[i]);
                else
                    writer.Write(new byte[] { 255, 255, 255 });
            }
        }

        private static void WriteGifFrame(BinaryWriter writer, byte[] pixels, int width, int height, int delay)
        {
            writer.Write((byte)0x21);
            writer.Write((byte)0xF9);
            writer.Write((byte)4);
            writer.Write((byte)0);
            WriteShort(writer, delay);
            writer.Write((byte)0);
            writer.Write((byte)0);

            writer.Write((byte)0x2C);
            WriteShort(writer, 0);
            WriteShort(writer, 0);
            WriteShort(writer, width);
            WriteShort(writer, height);
            writer.Write((byte)0);

            writer.Write((byte)2);

            byte[] compressed = GifSimpleEncode(pixels);

            int offset = 0;

            while (offset < compressed.Length)
            {
                int blockSize = Math.Min(255, compressed.Length - offset);
                writer.Write((byte)blockSize);
                writer.Write(compressed, offset, blockSize);
                offset += blockSize;
            }

            writer.Write((byte)0);
        }

        private static byte[] GifSimpleEncode(byte[] pixels)
        {
            List<int> codes = new List<int>();

            int clearCode = 4;
            int endCode = 5;

            foreach (byte pixel in pixels)
            {
                codes.Add(clearCode);
                codes.Add(pixel);
            }

            codes.Add(endCode);

            List<byte> bytes = new List<byte>();

            int bitBuffer = 0;
            int bitCount = 0;

            foreach (int code in codes)
            {
                bitBuffer |= code << bitCount;
                bitCount += 3;

                while (bitCount >= 8)
                {
                    bytes.Add((byte)(bitBuffer & 255));
                    bitBuffer >>= 8;
                    bitCount -= 8;
                }
            }

            if (bitCount > 0)
            {
                bytes.Add((byte)(bitBuffer & 255));
            }

            return bytes.ToArray();
        }
    }

    internal class Program
    {
        static void Main(string[] args)
        {
            Func<double[], double> sphereFunction = coords =>
            {
                double sum = 0;

                for (int i = 0; i < coords.Length; i++)
                {
                    sum += coords[i] * coords[i];
                }

                return sum;
            };

            double[] start = { 1.0, 1.5, 1.0 };

            NelderMeadOptimizer optimizer = new NelderMeadOptimizer(sphereFunction)
            {
                Step = 0.5,
                Accuracy = 1e-6,
                ReflectionCoeff = 1.0,
                ExpansionCoeff = 2.0,
                ContractionCoeff = 0.5,
                ShrinkCoeff = 0.5
            };

            double[] result = optimizer.Optimize(start, 500);

            Console.WriteLine("Найденный минимум:");

            for (int i = 0; i < result.Length; i++)
            {
                Console.WriteLine($"x{i + 1} = {result[i]:F6}");
            }

            Console.WriteLine($"Значение функции = {sphereFunction(result):F6}");

            SimplexVisualizer.SaveEvolutionPng(optimizer.History, "simplex_evolution.png");
            SimplexVisualizer.SaveAnimationGif(optimizer.History, "simplex_animation.gif");

            Console.WriteLine("Файл simplex_evolution.png создан.");
            Console.WriteLine("Файл simplex_animation.gif создан.");
        }
    }
}
