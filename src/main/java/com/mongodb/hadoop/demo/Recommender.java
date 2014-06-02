package com.mongodb.hadoop.demo;

import com.mongodb.hadoop.BSONFileInputFormat;
import com.mongodb.hadoop.MongoOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.storage.StorageLevel;
import org.bson.BSONObject;
import org.bson.BasicBSONObject;
import scala.Tuple2;

import java.util.Date;


public class Recommender {

    private static String MONGODB = "127.0.0.1:27017";
    private static String HDFS = "127.0.0.1:9000";

	public static void main(String[] args) {
        JavaSparkContext sc = new JavaSparkContext("local", "Recommender");

        Configuration bsonDataConfig = new Configuration();
        bsonDataConfig.set("mongo.job.input.format", "com.mongodb.hadoop.BSONFileInputFormat");

        Configuration predictionsConfig = new Configuration();
        predictionsConfig.set("mongo.output.uri", "mongodb://" + MONGODB + "/movielens.predictions");

        JavaPairRDD<Object,BSONObject> bsonRatingsData = sc.newAPIHadoopFile(
            "hdfs://" + HDFS + "/work/ratings.bson", BSONFileInputFormat.class, Object.class,
                BSONObject.class, bsonDataConfig);

        JavaRDD<Rating> ratingsData = bsonRatingsData.map(
            new Function<Tuple2<Object,BSONObject>,Rating>() {
                public Rating call(Tuple2<Object,BSONObject> doc) throws Exception {
                    Integer userid = (Integer) doc._2.get("userid");
                    Integer movieid = (Integer) doc._2.get("movieid");
                    Double rating = (Double) doc._2.get("rating");
                    return new Rating(userid, movieid, rating);
                }
            }
        );

        // keep this RDD in memory as much as possible, and spill to disk if needed
        ratingsData.persist(StorageLevel.MEMORY_AND_DISK());

        // create the model from existing ratings data
        MatrixFactorizationModel model = ALS.train(ratingsData.rdd(), 1, 20, 0.01);

        JavaRDD<Object> userData = sc.newAPIHadoopFile("hdfs://" + HDFS + "/work/users.bson",
                BSONFileInputFormat.class, Object.class, BSONObject.class, bsonDataConfig).map(
            new Function<Tuple2<Object, BSONObject>, Object>() {
                @Override
                public Object call(Tuple2<Object, BSONObject> doc) throws Exception {
                    return doc._2.get("userid");
                }
            }
        );

        JavaRDD<Object> movieData = sc.newAPIHadoopFile("hdfs://" + HDFS + "/work/movies.bson",
                BSONFileInputFormat.class, Object.class, BSONObject.class, bsonDataConfig).map(
            new Function<Tuple2<Object, BSONObject>, Object>() {
                @Override
                public Object call(Tuple2<Object, BSONObject> doc) throws Exception {
                    return doc._2.get("movieid");
                }
            }
        );

        // generate complete pairing for all possible (user,movie) combinations
        JavaPairRDD<Object,Object> usersMovies = userData.cartesian(movieData);

        // predict ratings
        JavaPairRDD<Object,BSONObject> predictions = model.predict(usersMovies.rdd()).toJavaRDD().mapToPair(
            new PairFunction<Rating, Object, BSONObject>() {
                @Override
                public Tuple2<Object, BSONObject> call(Rating rating) throws Exception {
                    BSONObject doc = new BasicBSONObject();
                    doc.put("userid", rating.user());
                    doc.put("movieid", rating.product());
                    doc.put("rating", rating.rating());
                    doc.put("timestamp", new Date());
                    // null key means an ObjectId will be generated on insert
                    return new Tuple2<Object, BSONObject>(null, doc);
                }
            }
        );

        predictions.saveAsNewAPIHadoopFile("file:///notapplicable",
            Object.class, Object.class, MongoOutputFormat.class, predictionsConfig);
	}
}
