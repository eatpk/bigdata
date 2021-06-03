package edu.gatech.cse6250.randomwalk

import edu.gatech.cse6250.model.{ PatientProperty, EdgeProperty, VertexProperty }
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
     * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
     * Return a List of patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay
     */

    /** Remove this placeholder and implement your code */
    val root: VertexId = patientID
    var processedGrph: Graph[Double, Double] = graph.outerJoinVertices(graph.outDegrees) { (vID, vData, optOutDegree) => optOutDegree.getOrElse(0) }
      .mapTriplets(edgeTriplets => 1.0 / edgeTriplets.srcAttr, TripletFields.Src)
      .mapVertices { (id, vData) => if ((id == root)) 1.0 else 0.0 }

    def isDifferent(src1: VertexId, src2: VertexId): Double = { if (src1 == src2) 1.0 else 0.0 }
    var i = 0
    var grph: Graph[Double, Double] = null
    while (i < numIter) {
      processedGrph.cache()
      val grp = processedGrph.aggregateMessages[Double](x => x.sendToDst(x.srcAttr * x.attr), _ + _, TripletFields.Src)
      grph = processedGrph
      val resetProb = { (vertexSource: VertexId, vertexDest: VertexId) => alpha * isDifferent(vertexSource, vertexDest) }
      processedGrph = processedGrph.outerJoinVertices(grp) { (x, y, z) => resetProb(root, x) + (1.0 - alpha) * z.getOrElse(0.0) }.cache()
      processedGrph.edges.foreachPartition(vertices => {})
      grph.vertices.unpersist(false)
      grph.edges.unpersist(false)
      i = i + 1
    }
    val randomWalk = graph.subgraph(vpred = (x, y) => y.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(x => x._1).collect().toSet
    val randomWalkFinal = processedGrph.vertices.filter(n => randomWalk.contains(n._1)).takeOrdered(11)(Ordering[Double].reverse.on(n => n._2)).map(_._1)
    randomWalkFinal.slice(1, randomWalkFinal.length).toList
  }
}
