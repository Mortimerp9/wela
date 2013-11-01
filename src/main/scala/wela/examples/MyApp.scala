package wela.examples

import wela.core._
import wela.classifiers._
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.LeastMedSq
import weka.classifiers.trees.RandomForest
import weka.classifiers.{Classifier => WekaClassifier}
import scalaz._
import Scalaz._

object MyApp extends App {


  val pbl = Problem("test", NominalAttribute('color, Seq('red, 'blue, 'green))) withAttributes(NumericAttribute('size),
    NumericAttribute('weight))

  val train = pbl withInstances(
    Instance(
      'size -> 10.0,
      'weight -> 10,
      'color -> 'blue),
    Instance(
      'size -> 11.0,
      'weight -> 10,
      'color -> 'red),
    Instance(
      'size -> 11.0,
      'weight -> 15,
      'color -> 'red),
    Instance(
      'size -> 11.0,
      'weight -> 20,
      'color -> 'red),
    Instance(
      'size -> 10.0,
      'weight -> 50,
      'color -> 'green),
    Instance(
      'size -> 10.0,
      'weight -> 55,
      'color -> 'green))

  val model = Classifier(new NaiveBayes()) train (train)
  val pred = model flatMap {
    cl =>
      cl.classifyInstance(Instance('size -> 10,
        'weight -> 50))
  }
  println(pred)

  val pbl2 = Problem("test2", NumericAttribute('size)) withAttributes(NominalAttribute('color, Seq('red, 'blue, 'green)),
    NumericAttribute('weight))

  val instan = Seq(
    Instance(
      'size -> 10.0,
      'weight -> 10,
      'color -> 'blue),
    Instance(
      'size -> 11.0,
      'weight -> 10,
      'color -> 'red),
    Instance(
      'size -> 11.0,
      'weight -> 15,
      'color -> 'red),
    Instance(
      'size -> 11.0,
      'weight -> 20,
      'color -> 'red),
    Instance(
      'size -> 10.0,
      'weight -> 50,
      'color -> 'green),
    Instance(
      'size -> 10.0,
      'weight -> 50,
      'color -> 'green))

  val train2 = pbl2 withInstances (instan: _*)

  val model3 = Classifier(new LeastMedSq()) train (train2)
  val pred3 = model3 flatMap {
    cl =>
      cl.classifyInstance(Instance('color -> 'red,
        'weight -> 50))
  }
  println(pred3)

  val train3 = pbl2.withMapping('color, NumericAttribute('color)) {
    case v: SymbolValue => v.name.length()
    case _ => 0
  }  withInstances (instan: _*)

  val model2 = Classifier(new LeastMedSq()) train (train3)
  val pred2 = model2 flatMap {
    cl =>
      cl.classifyInstance(Instance('color -> 'red,
        'weight -> 50))
  }
  println(pred2)

  //////////////////////////////////////////////  

  val txtInstances = Seq(Instance('text -> "liers say lies", 'truth -> 'false),
    Instance('text -> "what's false is a lie", 'truth -> 'false),
    Instance('text -> "I am a lier", 'truth -> 'false),
    Instance('text -> "this is a lie", 'truth -> 'false),
    Instance('text -> "this is true", 'truth -> 'true),
    Instance('text -> "true is good", 'truth -> 'true),
    Instance('text -> "liers don't say truth", 'truth -> 'true),
    Instance('text -> "what's true is true", 'truth -> 'true))

  val pbl3 = Problem("test processing", NominalAttribute('truth, Seq('true, 'false))).withAttributes(StringAttribute('text))

  val pbl4 = pbl3.withFlatMapping('text, Seq(NumericAttribute('lieCnt), NumericAttribute('truthCnt))) {
    case inst: StringValue =>
      val tokens = inst.split(" ")
      Seq('lieCnt -> tokens.count(_.equals("lie")),
        'truthCnt -> tokens.count(_.equals("true")))
    case _ => Nil
  }

  val train4 = pbl4 withInstances (txtInstances: _*)

  val model4 = Classifier(new NaiveBayes()) train (train4)
  val pred4 = model4 flatMap {
    cl =>
      cl.classifyInstance(Instance('text -> "true truth true"))
  }
  println(pred4)
  val pred5 = model4 flatMap {
    cl =>
      cl.classifyInstance(Instance('text -> "lie lie"))
  }
  println(pred5)

  ////////////////////////////

  val pbl5 = pbl3.explodeAttributes('text, "bow") {
    case inst: StringValue =>
      inst.split(" ").toSeq.map(_.toLowerCase.replaceAll("\\W", "_"))
    case _ => Nil
  }

  val train5 = pbl5.withInstances(txtInstances: _*)

  println(train5.mappedInstances)


  val model5 = Classifier(new NaiveBayes()) train (train5)
  val pred6 = model5 flatMap {
    cl =>
      cl.classifyInstance(Instance('text -> "true truth true"))
  }
  println(pred6)
  val pred7 = model5 flatMap {
    cl =>
      cl.classifyInstance(Instance('text -> "lie lie"))
  }
  println(pred7)

  //////////////////

  //use weka to do the serialization
  val fileName = "/tmp/model5.model"

  model5.foreach { cls =>
    weka.core.SerializationHelper.write(fileName, cls.cl)
  }

  val rawModel6 = weka.core.SerializationHelper.read(fileName).asInstanceOf[NaiveBayes]

  val newTrain = pbl5 withInstances()
  val rawInstance = newTrain.makeInstance(Instance('text -> "what's true is true"))
  val idx = rawModel6.classifyInstance(rawInstance)
  println(newTrain.problem.label.resolve(idx))

}