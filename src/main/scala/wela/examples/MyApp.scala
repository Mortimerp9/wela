package wela.examples

import wela.core._
import wela.classifiers._
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.LeastMedSq

object MyApp extends App {

  val pbl = Problem("test", NominalAttribute('color, Seq('red, 'blue, 'green))) withAttributes (NumericAttribute('size),
    NumericAttribute('weight))

  val train = pbl withInstances (
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
  val pred = model flatMap { cl =>
    cl.classifyInstance(Instance('size -> 10,
      'weight -> 50))
  }
  println(pred)

  val pbl2 = Problem("test2", NumericAttribute('size)) withAttributes (NominalAttribute('color, Seq('red, 'blue, 'green)),
    NumericAttribute('weight))

  val train2 = pbl2 withInstances (
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

  val model3 = Classifier(new LeastMedSq()) train (train2)
  val pred3 = model3 flatMap { cl =>
    cl.classifyInstance(Instance('color -> 'red,
      'weight -> 50))
  }
  println(pred3)

  val train3 = train2.withMapping('color, NumericAttribute('color)) {
    case v: NominalValue => NumericValue(v.value.name.length())
    case _ => NumericValue(0)
  }

  val model2 = Classifier(new LeastMedSq()) train (train3)
  val pred2 = model2 flatMap { cl =>
    cl.classifyInstance(Instance('color -> 'red,
      'weight -> 50))
  }
  println(pred2)

  val pbl3 = Problem("test processing", NominalAttribute('truth, Seq('true, 'false))) withAttributes (StringAttribute('text))

  val train4 = pbl3.withInstances(
    Instance('text -> "liers say lies", 'truth -> 'false),
    Instance('text -> "what's false is a lie", 'truth -> 'false),
    Instance('text -> "I am a lier", 'truth -> 'false),
    Instance('text -> "this is a lie", 'truth -> 'false),
    Instance('text -> "this is true", 'truth -> 'true),
    Instance('text -> "true is good", 'truth -> 'true),
    Instance('text -> "liers don't say truth", 'truth -> 'true),
    Instance('text -> "what's true is true", 'truth -> 'true)).withMapping('text, Seq(NumericAttribute('lieCnt), NumericAttribute('truthCnt))) {
    case inst: StringValue => 
      val tokens = inst.value.split(" ")
      Seq('lieCnt -> NumericValue(tokens.count(_.equals("lie"))),
        'truthCnt -> NumericValue(tokens.count(_.equals("true"))))
    case _ => Nil
  }

}