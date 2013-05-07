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
}