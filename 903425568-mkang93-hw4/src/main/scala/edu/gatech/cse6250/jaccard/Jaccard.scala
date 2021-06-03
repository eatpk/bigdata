/**
 *
 * students: please put your implementation in this file!
 */
package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */

    /** Remove this placeholder and implement your code */
    val patientVertices = graph.subgraph(vpred = (id, attr) => attr.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(x => x._1).collect().toSet
    val allVertices = graph.collectNeighborIds(EdgeDirection.Out)
    val otherPatients = allVertices.filter(f => patientVertices.contains(f._1) && f._1.toLong != patientID)
    val patientIDNeighbors = allVertices.filter(f => f._1.toLong == patientID).map(f => f._2).flatMap(f => f).collect().toSet
    val jaccards = otherPatients.map(f => (f._1, jaccard(patientIDNeighbors, f._2.toSet))).sortBy(x => x._2, false).take(10)
      .map(x => x._1.toLong)
      .toList

    jaccards
  }
  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
     * Given a patient, med, diag, lab graph, calculate pairwise similarity between all
     * patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where
     * patient-1-id < patient-2-id to avoid duplications
     */

    /** Remove this placeholder and implement your code */
    val sc = graph.edges.sparkContext
    val patientVertices = graph.subgraph(vpred = (id, attr) => attr.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(x => x._1).collect().toSet
    val allPatients = graph.collectNeighborIds(EdgeDirection.Out).filter(x => patientVertices.contains(x._1))
    val crossNeighbors = allPatients.cartesian(allPatients).filter(x => x._1._1 < x._2._1)
    crossNeighbors.map(x => (x._1._1, x._2._1, jaccard(x._1._2.toSet, x._2._2.toSet)))

  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */

    /** Remove this placeholder and implement your code */
    val s_ab = a.intersect(b).size.toDouble / a.union(b).size.toDouble

    if (s_ab.isNaN) 0.0 else s_ab
  }
}
