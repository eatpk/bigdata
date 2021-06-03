package edu.gatech.cse6250.phenotyping

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Sungtae An <stan84@gatech.edu>,
 */
object T2dmPhenotype {

  /** Hard code the criteria */
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648.0", "648.00", "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4", "250.*")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     * i.e. do NOT read from file or write file
     *
     * You don't need to follow the example placeholder code below exactly, but do have the same return type.
     *
     * Hint: Consider case sensitivity when doing string comparisons.
     */
    def abnormalLab(element: LabResult): Boolean = {

      var abnormal = false
      if (element.testName == "hba1c" && element.value >= 6.0) {
        abnormal = true
      } else if (element.testName == "hemoglobin a1c" && element.value >= 6.0) {
        abnormal = true
      } else if (element.testName == "fasting glucose" && element.value >= 110) {
        abnormal = true
      } else if (element.testName == "fasting blood glucose" && element.value >= 110) {
        abnormal = true
      } else if (element.testName == "fasting plasma glucose" && element.value >= 110) {
        abnormal = true
      } else if (element.testName == "glucose" && element.value > 110) {
        abnormal = true
      } else if (element.testName == "glucose" && element.value > 110) {
        abnormal = true
      } else if (element.testName == "glucose, serum" && element.value > 110) {
        abnormal = true
      }

      abnormal
    }
    val sc = medication.sparkContext
    val patients = diagnostic.map(x => x.patientID).union(labResult.map(x => x.patientID)).union(medication.map(x => x.patientID)).distinct()

    val t1dm = patients.subtract(diagnostic.filter(x => T1DM_DX.contains(x.code)).map(x => x.patientID)).distinct()
    val t2dm = diagnostic.filter(x => T2DM_DX.contains(x.code)).map(x => x.patientID).distinct()

    val t1med_no = patients.subtract(medication.filter(x => T1DM_MED.contains(x.medicine)).map(x => x.patientID)).distinct()
    val t1med_yes = medication.filter(x => T1DM_MED.contains(x.medicine)).map(x => x.patientID).distinct()
    val t2med_no = patients.subtract(medication.filter(x => T2DM_MED.contains(x.medicine)).map(x => x.patientID)).distinct()
    val t2med_yes = medication.filter(x => T2DM_MED.contains(x.medicine)).map(x => x.patientID).distinct()

    val third_pt = t1dm.intersection(t2dm).intersection(t1med_yes).intersection(t2med_yes).cache()
    val join_tuple = third_pt.map(x => (x, 0))
    val case_pt3_1 = medication.map(x => (x.patientID, x)).join(join_tuple).map(x => Medication(x._2._1.patientID, x._2._1.date, x._2._1.medicine))

    val case3_1 = case_pt3_1.filter(x => T1DM_MED.contains(x.medicine)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min(_, _))
    val case3_2 = case_pt3_1.filter(x => T2DM_MED.contains(x.medicine)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min(_, _))
    val case_pt3 = case3_1.join(case3_2).filter(x => (x._2._2 < x._2._1)).map(x => x._1).distinct()

    val case_pt1 = t1dm.intersection(t2dm).intersection(t1med_no).distinct()
    val case_pt2 = t1dm.intersection(t2dm).intersection(t1med_yes).intersection(t2med_no).distinct()

    val caseP = case_pt1.union(case_pt2).union(case_pt3)
    /** Find CASE Patients */
    val casePatients = caseP.map(x => (x, 1))

    val with_glucose = labResult.filter(x => x.testName.contains("glucose"))

    val normal_lab = with_glucose.filter(x => !abnormalLab(x)).map(_.patientID).distinct()

    val control_diag_filter = diagnostic.filter(x => DM_RELATED_DX.contains(x.code) || x.code.startsWith("250.")).map(_.patientID).distinct()

    val control = normal_lab.intersection(patients.subtract(control_diag_filter)).distinct()

    /** Find CONTROL Patients */
    val controlPatients = control.map(x => (x, 2))

    /** Find OTHER Patients */
    val others = patients.subtract(control.union(caseP)).map(x => (x, 3))

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)

    /** Return */
    phenotypeLabel
  }
}