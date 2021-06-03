/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    /** HINT: See Example of Making Patient Vertices Below */
    /** HINT: See Example of Making Patient Vertices Below */
    val vertexPatient: RDD[(VertexId, VertexProperty)] = patients
      .map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))
    val patientSize = vertexPatient.collect.toMap.size

    val vertexLabResultRDD = labResults.map(_.labName).distinct().zipWithIndex()
      .map(x => (x._1, x._2 + patientSize + 1))

    val vertexLabResult = vertexLabResultRDD.map(x => (x._2, LabResultProperty(x._1))).asInstanceOf[RDD[(VertexId, VertexProperty)]]
    val labResultSize = vertexLabResultRDD.collect.toMap.size

    val vertexMedRDD = medications.map(_.medicine).distinct().zipWithIndex()
      .map(x => (x._1, x._2 + labResultSize + patientSize + 1))

    val vertexMed = vertexMedRDD.map(x => (x._2, MedicationProperty(x._1))).asInstanceOf[RDD[(VertexId, VertexProperty)]]
    val medSize = vertexMedRDD.collect.toMap.size

    val vertexDiagRDD = diagnostics.map(_.icd9code).distinct().zipWithIndex().map(x => (x._1, x._2 + labResultSize + medSize + patientSize + 1))
    val vertexDiag = vertexDiagRDD.map(x => (x._2, DiagnosticProperty(x._1))).asInstanceOf[RDD[(VertexId, VertexProperty)]]

    val labMap = vertexLabResultRDD.collect.toMap
    val medMap = vertexMedRDD.collect.toMap
    val diagMap = vertexDiagRDD.collect.toMap

    /**
     * HINT: See Example of Making PatientPatient Edges Below
     *
     * This is just sample edges to give you an example.
     * You can remove this PatientPatient edges and make edges you really need
     */
    /*
    case class PatientPatientEdgeProperty(someProperty: SampleEdgeProperty) extends EdgeProperty
    val edgePatientPatient: RDD[Edge[EdgeProperty]] = patients
      .map({ p => Edge(p.patientID.toLong, p.patientID.toLong, SampleEdgeProperty("sample").asInstanceOf[EdgeProperty])
      })
    */
    val edgePatLab: RDD[Edge[EdgeProperty]] = labResults.map(x => ((x.patientID, x.labName), x)).reduceByKey { case (x, y) => if (x.date > y.date) x else y }.flatMap(x => List(Edge(x._1._1.toLong, labMap(x._1._2), PatientLabEdgeProperty(x._2)), Edge(labMap(x._1._2), x._1._1.toLong, PatientLabEdgeProperty(x._2))))
    val edgePatMed: RDD[Edge[EdgeProperty]] = medications.map(x => ((x.patientID, x.medicine), x)).reduceByKey { case (x, y) => if (x.date > y.date) x else y }.flatMap(x => List(Edge(x._1._1.toLong, medMap(x._1._2), PatientMedicationEdgeProperty(x._2)), Edge(medMap(x._1._2), x._1._1.toLong, PatientMedicationEdgeProperty(x._2))))
    val edgePatDiag: RDD[Edge[EdgeProperty]] = diagnostics.map(x => ((x.patientID, x.icd9code), x)).reduceByKey { case (x, y) => if (x.date > y.date) x else y }.flatMap(x => List(Edge(x._1._1.toLong, diagMap(x._1._2), PatientDiagnosticEdgeProperty(x._2)), Edge(diagMap(x._1._2), x._1._1.toLong, PatientDiagnosticEdgeProperty(x._2))))

    val vertices = vertexPatient.union(vertexLabResult).union(vertexMed).union(vertexDiag)

    val edgePatientPatient = edgePatLab.union(edgePatDiag).union(edgePatMed)

    val graph: Graph[VertexProperty, EdgeProperty] = Graph[VertexProperty, EdgeProperty](vertices, edgePatientPatient)

    graph
  }
}
