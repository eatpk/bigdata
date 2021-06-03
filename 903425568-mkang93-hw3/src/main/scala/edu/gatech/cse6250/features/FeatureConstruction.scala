package edu.gatech.cse6250.features

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su
 */
object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   *
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val diagnostic_FT = diagnostic.map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)
    diagnostic_FT
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   *
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val medication_FT = medication.map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)
    medication_FT
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   *
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val labResult_FT_cnt = labResult.map(x => ((x.patientID, x.testName), 1.0)).reduceByKey(_ + _)
    val labResult_FT_val = labResult.map(x => ((x.patientID, x.testName), x.value)).reduceByKey(_ + _)
    val labResult_FT = labResult_FT_val.join(labResult_FT_cnt).map(x => (x._1, x._2._1 / x._2._2))
    labResult_FT

  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   *
   * @param diagnostic   RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val diag_candFT = diagnostic.filter(x => candiateCode.contains(x.code)).map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)

    diag_candFT
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   *
   * @param medication          RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val medication_candFT = medication.filter(x => candidateMedication.contains(x.medicine)).map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)

    medication_candFT
  }

  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   *
   * @param labResult    RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val lab_candFT = labResult.filter(x => candidateLab.contains(x.testName)).map(x => ((x.patientID, x.testName), x.value)).aggregateByKey((0.0, 0.0))(
      (acc, x) => (acc._1 + x, acc._2 + 1.0),
      (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2)).mapValues(x => x._1 / x._2) //acc._1 sum value, acc._2 count by key

    lab_candFT
  }

  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   *
   * @param sc      SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    val featureMap = feature.map(x => x._1._2).distinct().collect.zipWithIndex.toMap
    /**
     * Functions maybe helpful:
     * collect
     * groupByKey
     */
    val patientAndFeatures = feature.map(x => (x._1._1, (x._1._2, x._2))).groupByKey()

    val result = patientAndFeatures.map {
      case (target, features) =>
        val num_features = featureMap.size
        val mapped_features = features.toList.map { case (featureName, featureValue) => (featureMap(featureName).toInt, featureValue) }

        val vec = Vectors.sparse(num_features, mapped_features)
        val vectorize = (target, vec)
        vectorize
    }
    result

    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */

  }
}

