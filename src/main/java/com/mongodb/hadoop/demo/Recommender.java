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

    private static String MONGODB_HOST = "127.0.0.1:27017";
    private static String HDFS_HOST = "127.0.0.1:9000";

    private static String MONGODB_OUTPUT = "movielens.predictions";
    private static String RATINGS_FILE = "/work/movielens-small/ratings.bson";
    private static String USERS_FILE = "/work/movielens-small/users.bson";
    private static String MOVIES_FILE = "/work/movielens/movies.bson";

    private static String MONGODB_OUTPUT_URI = "mongodb://" + MONGODB_HOST + MONGODB_OUTPUT;
    private static String RATINGS_URI = "hdfs://" + HDFS_HOST + RATINGS_FILE;
    private static String USERS_URI = "hdfs://" + HDFS_HOST + USERS_FILE;
    private static String MOVIES_URI = "hdfs://" + HDFS_HOST + MOVIES_FILE;

	public static void main(String[] args) {
        JavaSparkContext sc = new JavaSparkContext("local", "Recommender");

        Configuration bsonDataConfig = new Configuration();
        bsonDataConfig.set("mongo.job.input.format", "com.mongodb.hadoop.BSONFileInputFormat");

        Configuration predictionsConfig = new Configuration();
        predictionsConfig.set("mongo.output.uri", MONGODB_OUTPUT_URI);

        JavaPairRDD<Object,BSONObject> bsonRatingsData = sc.newAPIHadoopFile(
            RATINGS_URI, BSONFileInputFormat.class, Object.class,
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

        JavaRDD<Object> userData = sc.newAPIHadoopFile(USERS_URI,
                BSONFileInputFormat.class, Object.class, BSONObject.class, bsonDataConfig).map(
            new Function<Tuple2<Object, BSONObject>, Object>() {
                @Override
                public Object call(Tuple2<Object, BSONObject> doc) throws Exception {
                    return doc._2.get("userid");
                }
            }
        );

        JavaRDD<Object> movieData = sc.newAPIHadoopFile(MOVIES_URI,
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
