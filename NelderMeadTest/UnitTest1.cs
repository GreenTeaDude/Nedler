using System;
using Xunit;
using NelderMeadMethod.Core;
using NelderMeadMethod.Optimization;

namespace NelderMeadMethod.Tests
{
    public class NelderMeadTests
    {
        [Fact]
        public void Simplex_CreatesCorrectNumberOfVertices()
        {
            double[] startPoint = { 1.0, 2.0, 3.0 };
            double step = 0.5;

            Simplex simplex = new Simplex(startPoint, step);

            Assert.Equal(3, simplex.Dimension);
            Assert.Equal(4, simplex.VertexCount);
            Assert.Equal(4, simplex.Vertexes.Length);
            Assert.Equal(4, simplex.Values.Length);
        }

        [Fact]
        public void Simplex_FirstVertex_EqualsStartPoint()
        {
            double[] startPoint = { 1.0, 2.0, 3.0 };
            Simplex simplex = new Simplex(startPoint, 0.5);

            Assert.Equal(startPoint, simplex.Vertexes[0]);
        }

        [Fact]
        public void Simplex_OtherVertices_AreShiftedByStep()
        {
            double[] startPoint = { 1.0, 2.0, 3.0 };
            double step = 0.5;

            Simplex simplex = new Simplex(startPoint, step);

            Assert.Equal(new double[] { 1.5, 2.0, 3.0 }, simplex.Vertexes[1]);
            Assert.Equal(new double[] { 1.0, 2.5, 3.0 }, simplex.Vertexes[2]);
            Assert.Equal(new double[] { 1.0, 2.0, 3.5 }, simplex.Vertexes[3]);
        }

        [Fact]
        public void UpdateValues_ComputesFunctionValuesCorrectly()
        {
            double[] startPoint = { 1.0, 2.0 };
            Simplex simplex = new Simplex(startPoint, 1.0);

            Func<double[], double> sphere = coords =>
            {
                double sum = 0;
                foreach (double x in coords)
                    sum += x * x;
                return sum;
            };

            simplex.UpdateValues(sphere);

            Assert.Equal(5.0, simplex.Values[0], 6);
            Assert.Equal(8.0, simplex.Values[1], 6);
            Assert.Equal(10.0, simplex.Values[2], 6);
        }

        [Fact]
        public void SortVertexes_SortsValuesAscending()
        {
            double[] startPoint = { 1.0, 2.0 };
            Simplex simplex = new Simplex(startPoint, 1.0);

            Func<double[], double> function = coords => coords[0] + coords[1];

            simplex.UpdateValues(function);
            simplex.SortVertexes();

            Assert.True(simplex.Values[0] <= simplex.Values[1]);
            Assert.True(simplex.Values[1] <= simplex.Values[2]);
        }

        [Fact]
        public void CalculateCentroid_ReturnsCorrectCentroidWithoutWorstVertex()
        {
            double[] startPoint = { 0.0, 0.0 };
            Simplex simplex = new Simplex(startPoint, 1.0);

            simplex.Vertexes[0] = new double[] { 0.0, 0.0 };
            simplex.Vertexes[1] = new double[] { 0.0, 1.0 };
            simplex.Vertexes[2] = new double[] { 1.0, 0.0 };

            simplex.Values[0] = 0.0;
            simplex.Values[1] = 1.0;
            simplex.Values[2] = 2.0;

            double[] centroid = simplex.CalculateCentroid();

            Assert.Equal(0.0, centroid[0], 6);
            Assert.Equal(0.5, centroid[1], 6);
        }

        [Fact]
        public void IsConverged_ReturnsTrue_WhenDifferenceLessThanAccuracy()
        {
            double[] startPoint = { 0.0, 0.0 };
            Simplex simplex = new Simplex(startPoint, 1.0);

            simplex.Values[0] = 1.000000;
            simplex.Values[1] = 1.0000004;
            simplex.Values[2] = 1.0000008;

            Assert.True(simplex.IsConverged(1e-5));
        }

        [Fact]
        public void IsConverged_ReturnsFalse_WhenDifferenceGreaterThanAccuracy()
        {
            double[] startPoint = { 0.0, 0.0 };
            Simplex simplex = new Simplex(startPoint, 1.0);

            simplex.Values[0] = 1.0;
            simplex.Values[1] = 1.5;
            simplex.Values[2] = 2.0;

            Assert.False(simplex.IsConverged(1e-5));
        }

        [Fact]
        public void Optimize_SphereFunction_DecreasesObjectiveValue()
        {
            Func<double[], double> sphere = coords =>
            {
                double sum = 0;
                foreach (double x in coords)
                    sum += x * x;
                return sum;
            };

            double[] start = { 1.0, 1.5, 1.0 };

            NelderMeadOptimizer optimizer = new NelderMeadOptimizer(sphere)
            {
                Step = 0.5,
                Accuracy = 1e-6,
                ReflectionCoeff = 1.0,
                ExpansionCoeff = 2.0,
                ContractionCoeff = 0.5,
                ShrinkCoeff = 0.5
            };

            double[] result = optimizer.Optimize(start, 500);

            double startValue = sphere(start);
            double resultValue = sphere(result);

            Assert.True(resultValue < startValue);
        }

        [Fact]
        public void Optimize_SphereFunction_ResultIsCloseToZero()
        {
            Func<double[], double> sphere = coords =>
            {
                double sum = 0;
                foreach (double x in coords)
                    sum += x * x;
                return sum;
            };

            double[] start = { 1.0, 1.5, 1.0 };

            NelderMeadOptimizer optimizer = new NelderMeadOptimizer(sphere)
            {
                Step = 0.5,
                Accuracy = 1e-6,
                ReflectionCoeff = 1.0,
                ExpansionCoeff = 2.0,
                ContractionCoeff = 0.5,
                ShrinkCoeff = 0.5
            };

            double[] result = optimizer.Optimize(start, 500);

            Assert.InRange(result[0], -0.01, 0.01);
            Assert.InRange(result[1], -0.01, 0.01);
            Assert.InRange(result[2], -0.01, 0.01);
        }

        [Fact]
        public void Optimize_ThrowsException_WhenMaxIterationsInvalid()
        {
            Func<double[], double> sphere = coords => coords[0] * coords[0];
            NelderMeadOptimizer optimizer = new NelderMeadOptimizer(sphere);

            Assert.Throws<ArgumentOutOfRangeException>(() => optimizer.Optimize(new double[] { 1.0 }, 0));
        }

        [Fact]
        public void Constructor_ThrowsException_WhenStepInvalid()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new Simplex(new double[] { 1.0, 2.0 }, 0));
        }
    }
}