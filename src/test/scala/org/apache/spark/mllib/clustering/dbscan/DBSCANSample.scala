package org.apache.spark.mllib.clustering.dbscan

import breeze.numerics.log
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Create by NING on 2018/6/20.<br>
  *
  */
object DBSCANSample {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("DBSCAN Sample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("D:\\labeled_data.csv")

    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
    //parsedData.collect().foreach(Vector => println(Vector))

    //log.info(s"EPS: $eps minPoints: $minPoints")

    val model = DBSCAN.train(
      parsedData,
      //eps = eps,
      eps = 0.1,
      //minPoints = minPoints,
      minPoints = 3,
      //maxPointsPerPartition = maxPointsPerPartition,
      maxPointsPerPartition = 400)


    model.labeledPoints.map(p => s"${p.x},${p.y},${p.cluster}").saveAsTextFile("D:\\labeled_data_result")

    sc.stop()
  }

}
