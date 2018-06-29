/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.mllib.clustering.dbscan

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.PairRDDFunctions
import org.apache.spark.mllib.clustering.dbscan.DBSCANLabeledPoint.Flag
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Top level method for calling DBSCAN
  */
object DBSCAN {

  /**
    * Train a DBSCAN Model using the given set of parameters
    *
    * @param data                  training points stored as `RDD[Vector]`
    *                              only the first two points of the vector are taken into consideration
    * @param eps                   the maximum distance between two points for them to be considered as part
    *                              of the same region
    * @param minPoints             the minimum number of points required to form a dense region
    * @param maxPointsPerPartition the largest number of points in a single partition
    */
  def train(
             data: RDD[Vector],
             eps: Double,
             minPoints: Int,
             maxPointsPerPartition: Int): DBSCAN = {

    new DBSCAN(eps, minPoints, maxPointsPerPartition, null, null).train(data)

  }

}

/**
  * A parallel implementation of DBSCAN clustering. The implementation will split the data space
  * into a number of partitions, making a best effort to keep the number of points in each
  * partition under `maxPointsPerPartition`. After partitioning, traditional DBSCAN
  * clustering will be run in parallel for each partition and finally the results
  * of each partition will be merged to identify global clusters.
  *
  * This is an iterative algorithm that will make multiple passes over the data,
  * any given RDDs should be cached by the user.
  */
class DBSCAN private(
                      val eps: Double,
                      val minPoints: Int,
                      val maxPointsPerPartition: Int,
                      @transient val partitions: List[(Int, DBSCANRectangle)],
                      @transient private val labeledPartitionedPoints: RDD[(Int, DBSCANLabeledPoint)])

  extends Serializable with Logging {
  type Margins = (DBSCANRectangle, DBSCANRectangle, DBSCANRectangle)

  private def train(vectors: RDD[Vector]): DBSCAN = {
    //    println("================vectors==================")
    //    vectors.collect().take(5).foreach(Vector => println(Vector))
    //    vectors=[0.6522709149272675,2.001981395200717,1.0]
    //    [-1.6059412399083612,-0.765224117959865,3.0]
    //    [-0.7304943294502382,-0.4843781490647655,3.0]
    //    [1.3270827261186386,-0.34507151484323034,2.0]
    //    [0.621920898963074,1.9737597649634844,1.0]

    // generate the smallest rectangles that split the space
    // and count how many points are contained in each one of them

    //    val minimumRectanglesWithCount1 = vectors.map(toMinimumBoundingRectangle)
    //    minimumRectanglesWithCount1.collect().foreach(u => println(u))
    //    val minimumRectanglesWithCount2 = minimumRectanglesWithCount1.map((_, 1))
    //    minimumRectanglesWithCount2.collect().foreach(u1 =>println(u1))

    //minimumRectanglesWithCount=Set((DBSCANRectangle(1.2,-1.7999999999999998,1.7999999999999998,-1.1999999999999997),1))
    //注意：minimumRectanglesWithCount集合中每一个元素的最后一个分量1已经不是测试数据中自带的1了
    val minimumRectanglesWithCount =
      vectors
        .map(toMinimumBoundingRectangle)
        .map((_, 1))
        .aggregateByKey(0)(_ + _, _ + _)
        .collect()
        .toSet


    //    println("===============minimumRectanglesWithCount===============")
    //    println(minimumRectanglesWithCount)

    // find the best partitions for the data space
    //localPartitions=(DBSCANRectangle(-2.4,-2.4,2.4,2.4),749)
    val localPartitions = EvenSplitPartitioner
      .partition(minimumRectanglesWithCount, maxPointsPerPartition, minimumRectangleSize)

    //localPartitions.foreach(u=>println(u))

    //这里应该是打印所有点所在的矩形区域，(DBSCANRectangle(-2.4,-2.4,2.4,2.4),749)
    logDebug("Found partitions: ")
    localPartitions.foreach(p => logDebug(p.toString))

    // grow partitions to include eps
    //把半径加入到分区当中
    val localMargins =
    localPartitions
      //这里map成三个矩形，第一个p.shrink(eps)在最里面，p.shrink(-eps)在最外面
      .map({ case (p, _) => (p.shrink(eps), p, p.shrink(-eps)) })
      //zipWithIndex为数组创建索引，从0开始，这里的索引打在每个partition的后面，看结果
      .zipWithIndex
    //localMargins=((DBSCANRectangle(-2.1,-2.1,2.1,2.1),DBSCANRectangle(-2.4,-2.4,2.4,2.4),DBSCANRectangle(-2.6999999999999997,-2.6999999999999997,2.6999999999999997,2.6999999999999997)),0)
    //localMargins.foreach(u => println(u.toString()))

    //将localMargins设置为广播变量
    val margins = vectors.context.broadcast(localMargins)

    // assign each point to its proper partition
    //将每个点分配到最恰当的区域，并为每个点打上分区号，分区号在前面
    //duplicated=DBSCANPoint([0.6522709149272675,2.001981395200717,1.0])-->0
    //duplicated=DBSCANPoint([-1.6059412399083612,-0.765224117959865,3.0])-->0
    val duplicated = for {
      //注意这里的vectors，它把[-1.6059412399083612,-0.765224117959865,3.0]引回来了
      point <- vectors.map(DBSCANPoint)
      ((inner, main, outer), id) <- margins.value
      if outer.contains(point)
    } yield (id, point)

    duplicated.collect().foreach(u =>println(u._2 + "-->" + u._1) )

    val numOfPartitions = localPartitions.size

    // 在每个partition本地执行dbscan
    //clustered=[0.6522709149272675,2.001981395200717,1.0],1,Core-->0
    //[-1.6059412399083612,-0.765224117959865,3.0],2,Core-->0
    //[-0.7304943294502382,-0.4843781490647655,3.0],2,Core-->0
    //[1.3270827261186386,-0.34507151484323034,2.0],3,Core-->0
    //[0.621920898963074,1.9737597649634844,1.0],1,Core-->0
    //==================================================================================================
    val clustered =
    duplicated
      .groupByKey(numOfPartitions)
      .flatMapValues(points =>
        new LocalDBSCANNaive(eps, minPoints).fit(points))
      .cache()
    //clustered.collect().foreach(u =>println(u._2 + "-->" + u._1) )

    // find all candidate points for merging clusters and group them
    //找出所有待合并类别的候选点，并对它们进行分组
    //mergePoints=(0,CompactBuffer((0,[-0.13447918058321057,2.110397484451179,0.0],0,Noise), (0,[-2.142194802916965,-0.4822720767833232,3.0],2,Border)
    val mergePoints =
    clustered
      .flatMap({
        case (partition, point) =>
          margins.value
            .filter({
              case ((inner, main, _), _) => main.contains(point) && !inner.almostContains(point)
            })
            .map({
              case (_, newPartition) => (newPartition, (partition, point))
            })
      })
      .groupByKey()
    //mergePoints.collect().foreach(u => println(u.toString()))

    logDebug("About to find adjacencies")
    // find all clusters with aliases from merging candidates
    //找出所有的类别
    val adjacencies =
    mergePoints
      .flatMapValues(findAdjacencies)
      .values
      .collect()
    //adjacencies.foreach(u => println(u.toString()))

    // generated adjacency graph
    val adjacencyGraph = adjacencies.foldLeft(DBSCANGraph[ClusterId]()) {
      case (graph, (from, to)) => graph.connect(from, to)
    }

    logDebug("About to find all cluster ids")
    // find all cluster ids
    //找出所有的类别ID
    val localClusterIds =
    clustered
      .filter({ case (_, point) => point.flag != Flag.Noise })
      .mapValues(_.cluster)
      .distinct()
      .collect()
      .toList

    localClusterIds.foreach(u => println(u.toString()))

    // assign a global Id to all clusters, where connected clusters get the same id
    //为所有的类别分配一个全局的类别ID
    val (total, clusterIdToGlobalId) = localClusterIds.foldLeft((0, Map[ClusterId, Int]())) {
      case ((id, map), clusterId) => {

        map.get(clusterId) match {
          case None => {
            val nextId = id + 1
            val connectedClusters = adjacencyGraph.getConnected(clusterId) + clusterId
            logDebug(s"Connected clusters $connectedClusters")
            val toadd = connectedClusters.map((_, nextId)).toMap
            (nextId, map ++ toadd)
          }
          case Some(x) =>
            (id, map)
        }

      }
    }

    logDebug("Global Clusters")
    clusterIdToGlobalId.foreach(e => logDebug(e.toString))
    logInfo(s"Total Clusters: ${localClusterIds.size}, Unique: $total")

    val clusterIds = vectors.context.broadcast(clusterIdToGlobalId)

    logDebug("About to relabel inner points")
    // relabel non-duplicated points
    val labeledInner =
      clustered
        .filter(isInnerPoint(_, margins.value))
        .map {
          case (partition, point) => {

            if (point.flag != Flag.Noise) {
              point.cluster = clusterIds.value((partition, point.cluster))
            }

            (partition, point)
          }
        }

    logDebug("About to relabel outer points")
    // de-duplicate and label merge points
    val labeledOuter =
      mergePoints.flatMapValues(partition => {
        partition.foldLeft(Map[DBSCANPoint, DBSCANLabeledPoint]())({
          case (all, (partition, point)) =>

            if (point.flag != Flag.Noise) {
              point.cluster = clusterIds.value((partition, point.cluster))
            }

            all.get(point) match {
              case None => all + (point -> point)
              case Some(prev) => {
                // override previous entry unless new entry is noise
                if (point.flag != Flag.Noise) {
                  prev.flag = point.flag
                  prev.cluster = point.cluster
                }
                all
              }
            }

        }).values
      })

    val finalPartitions = localMargins.map {
      case ((_, p, _), index) => (index, p)
    }

    logDebug("Done")

    new DBSCAN(
      eps,
      minPoints,
      maxPointsPerPartition,
      finalPartitions,
      labeledInner.union(labeledOuter))

  }

  type ClusterId = (Int, Int)

  def minimumRectangleSize = 2 * eps

  def labeledPoints: RDD[DBSCANLabeledPoint] = {
    labeledPartitionedPoints.values
  }

  /**
    * Find the appropriate label to the given `vector`
    *
    * This method is not yet implemented
    */
  def predict(vector: Vector): DBSCANLabeledPoint = {
    throw new NotImplementedError
  }

  private def isInnerPoint(
                            entry: (Int, DBSCANLabeledPoint),
                            margins: List[(Margins, Int)]): Boolean = {
    entry match {
      case (partition, point) =>
        val ((inner, _, _), _) = margins.filter({
          case (_, id) => id == partition
        }).head

        inner.almostContains(point)
    }
  }

  private def findAdjacencies(
                               partition: Iterable[(Int, DBSCANLabeledPoint)]): Set[((Int, Int), (Int, Int))] = {

    val zero = (Map[DBSCANPoint, ClusterId](), Set[(ClusterId, ClusterId)]())

    val (seen, adjacencies) = partition.foldLeft(zero)({

      case ((seen, adjacencies), (partition, point)) =>

        // noise points are not relevant for adjacencies
        if (point.flag == Flag.Noise) {
          (seen, adjacencies)
        } else {

          val clusterId = (partition, point.cluster)

          seen.get(point) match {
            case None => (seen + (point -> clusterId), adjacencies)
            case Some(prevClusterId) => (seen, adjacencies + ((prevClusterId, clusterId)))
          }

        }
    })

    adjacencies
  }

  //这里返回的DBSCANRectangle已经把vector的最后一个分量给去掉了
  private def toMinimumBoundingRectangle(vector: Vector): DBSCANRectangle = {
    val point = DBSCANPoint(vector)
    val x = corner(point.x)
    val y = corner(point.y)
    DBSCANRectangle(x, y, x + minimumRectangleSize, y + minimumRectangleSize)
  }

  private def corner(p: Double): Double =
    (shiftIfNegative(p) / minimumRectangleSize).intValue * minimumRectangleSize

  private def shiftIfNegative(p: Double): Double =
    if (p < 0) p - minimumRectangleSize else p

}
