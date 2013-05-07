package wela.examples

import wela.core._
import wela.classifiers._
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.LeastMedSq

object MyApp extends App {

  val pbl = Problem("test", Some('color)) withAttributes (NumericAttribute('size),
    NominalAttribute('color, Seq('red, 'blue, 'green)),
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
                'weight -> 50,
                'color -> 'green))

  val model = Classifier(new NaiveBayes()) train (train)
  val pred = model map { cl =>
    cl.classifyInstance(Instance('size -> 10,
      'weight -> 50))
  }
  println(pred)
  
  
  val pbl2 = Problem("test2", Some('size)) withAttributes (NumericAttribute('size),
    NominalAttribute('color, Seq('red, 'blue, 'green)),
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

  val model2 = Classifier(new LeastMedSq()) train (train2)
  val pred2 = model2 map { cl =>
    cl.classifyInstance(Instance('color -> 'red,
      'weight -> 50))
  }
  println(pred2)
}