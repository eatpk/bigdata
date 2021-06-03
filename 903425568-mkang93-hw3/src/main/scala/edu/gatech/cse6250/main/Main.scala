package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.clustering.Metrics
import edu.gatech.cse6250.features.FeatureConstruction
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import edu.gatech.cse6250.phenotyping.T2dmPhenotype
import org.apache.spark.mllib.clustering.{ GaussianMixture, KMeans, StreamingKMeans }
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{ DenseMatrix, Matrices, Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.io.Source

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Yu Jing <yjing43@gatech.edu>,
 * @author Ming Liu <mliu302@gatech.edu>
 */
object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext
    //  val sqlContext = spark.sqlContext

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(spark)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication))

    // =========== USED FOR AUTO GRADING CLUSTERING GRADING =============
    //phenotypeLabel.map { case (a, b) => s"$a\t$b" }.saveAsTextFile("data/phenotypeLabel")
    //featureTuples.map { case ((a, b), c) => s"$a\t$b\t$c" }.saveAsTextFile("data/featureTuples")
    // return
    // ==================================================================

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, streamingPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKmeans is: $streamingPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication))

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, streamingPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of StreamingKmeans is: $streamingPurity2%.5f")
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures: RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    println("phenotypeLabel: " + phenotypeLabel.count)
    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray))) })
    println("features: " + features.count)
    val rawFeatureVectors = features.map(_._2).cache()
    println("rawFeatureVectors: " + rawFeatureVectors.count)

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]

    def transform(feature: Vector): Vector = {
      val scaled = scaler.transform(Vectors.dense(feature.toArray))
      Vectors.dense(Matrices.dense(1, scaled.size, scaled.toArray).multiply(densePc).toArray)
    }

    /**
     * TODO: K Means Clustering using spark mllib
     * Train a k means model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder bow after your implementation
     */
    val k_means = new KMeans().setSeed(6250L).setK(3).setMaxIterations(20).run(featureVectors).predict(featureVectors)
    val k_means_pred = features.map(x => x._1).zip(k_means).join(phenotypeLabel).map(x => x._2)
    val kMeansPurity = Metrics.purity(k_means_pred)
    //val forpt = k_means_pred.map(x => (x, 1)).reduceByKey(_ + _)
    //forpt.collect().foreach(println)

    /**
     * TODO: GMMM Clustering using spark mllib
     * Train a Gaussian Mixture model using the variabe featureVectors as input
     * Set maxIterations =20 and seed as 6250L
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    val gaussian = new GaussianMixture().setSeed(6250L).setK(3).setMaxIterations(20).run(featureVectors).predict(featureVectors)
    val gaussian_pred = features.map(x => x._1).zip(gaussian).join(phenotypeLabel).map(x => x._2)
    val gaussianMixturePurity = Metrics.purity(gaussian_pred)

    //val forptg = gaussian_pred.map(x => (x, 1)).reduceByKey(_ + _).collect().foreach(println)
    /**
     * TODO: StreamingKMeans Clustering using spark mllib
     * Train a StreamingKMeans model using the variabe featureVectors as input
     * Set the number of cluster K = 3, DecayFactor = 1.0, number of dimensions = 10, weight for each center = 0.5, seed as 6250L
     * In order to feed RDD[Vector] please use latestModel, see more info: https://spark.apache.org/docs/2.2.0/api/scala/index.html#org.apache.spark.mllib.clustering.StreamingKMeans
     * To run your model, set time unit as 'points'
     * Assign each feature vector to a cluster(predicted Class)
     * Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
     * Find Purity using that RDD as an input to Metrics.purity
     * Remove the placeholder below after your implementation
     */
    val streamKmeans = new StreamingKMeans().setK(3).setDecayFactor(1.0).setRandomCenters(10, 0.5, 6250L).latestModel()
    val streamKmeans_pred = streamKmeans.update(featureVectors, 1.0, "points").predict(featureVectors)
    val streamKmeans_test = features.map(_._1).zip(streamKmeans_pred).join(phenotypeLabel).map(_._2)
    val streamKmeansPurity = Metrics.purity(streamKmeans_test)

    //val forpts = streamKmeans_test.map(x => (x, 1)).reduceByKey(_ + _).collect().foreach(println)

    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity)
  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
   *
   * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd'T'HH:mm:ssX"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def loadRddRawData(spark: SparkSession): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /* the sql queries in spark required to import sparkSession.implicits._ */
    import spark.implicits._
    val sqlContext = spark.sqlContext

    /* a helper function sqlDateParser may useful here */

    /**
     * load data using Spark SQL into three RDDs and return them
     * Hint:
     * You can utilize edu.gatech.cse6250.helper.CSVHelper
     * through your sparkSession.
     *
     * This guide may helps: https://bit.ly/2xnrVnA
     *
     * Notes:Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type.
     * Be careful when you deal with String and numbers in String type.
     * Ignore lab results with missing (empty or NaN) values when these are read in.
     * For dates, use Date_Resulted for labResults and Order_Date for medication.
     *
     */

    val medication_input = CSVHelper.loadCSVAsTable(spark, "data/medication_orders_INPUT.csv", "med_table")
    val medication_DF = spark.sql("select Member_ID, Order_Date, Drug_Name from med_table")

    val lab_input = CSVHelper.loadCSVAsTable(spark, "data/lab_results_INPUT.csv", "lab_table")
    val lab_DF = spark.sql("select Member_ID, Date_resulted, Result_Name,Numeric_Result from lab_table where Numeric_Result != '' AND Numeric_Result IS NOT NULL ")

    val encounter_input = CSVHelper.loadCSVAsTable(spark, "data/encounter_INPUT.csv", "encounter_table")
    val encounter_dx_input = CSVHelper.loadCSVAsTable(spark, "data/encounter_dx_INPUT.csv", "encounter_dx_table")

    val diag_DF = spark.sql("select encounter_table.Member_ID, encounter_table.Encounter_DateTime, encounter_dx_table.code from encounter_table join encounter_dx_table on encounter_table.Encounter_ID = encounter_dx_table.Encounter_ID")

    /*
    case class Diagnostic(patientID: String, date: Date, code: String)

    case class LabResult(patientID: String, date: Date, testName: String, value: Double)

    case class Medication(patientID: String, date: Date, medicine: String)
    */
    val medication: RDD[Medication] = medication_DF.map(x => Medication(x(0).toString, sqlDateParser(x(1).toString), x(2).toString.toLowerCase)).rdd
    val labResult: RDD[LabResult] = lab_DF.map(x => LabResult(x(0).toString, sqlDateParser(x(1).toString), x(2).toString.toLowerCase, x(3).toString.filterNot(",".toSet).toDouble)).rdd
    val diagnostic: RDD[Diagnostic] = diag_DF.map(x => Diagnostic(x(0).toString, sqlDateParser(x(1).toString), x(2).toString)).rdd

    (medication, labResult, diagnostic)
  }

}
