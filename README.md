wela
====

A Scala wrapper around [Weka Machine](http://www.cs.waikato.ac.nz/ml/weka) Learning java library.

This is currently a very preliminary development. It is just a wrapper around the main classification features that I personally need to help perform predictions from scala code.

The current code already tries to simplify the use of Weka by adding more type safety and making the code more straightforward to code. As it's a light wrapper around Weka features, even if it tries to make things more immutable and "pure", there are loads of scary bits in it.

Usage
-----

The scala syntax already simplifies the use of Weka in your core quite a lot. This is an example based on the [Weka Programmatic Use example](http://weka.wikispaces.com/Programmatic+Use):

1. Create a problema define the prediction label and dataset attributes:

```scala
  val pbl = Problem("test", NominalAttribute('color, Seq('red, 'blue, 'green))) withAttributes(NumericAttribute('size), NumericAttribute('weight))
```

2. Load some data for the problem:

```scala
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
```

3. Train a Classifier

```scala
  val model = Classifier(new NaiveBayes()) train (train)
```

4. Try to predict the label of a new instance (`model` is an `Option`)

```scala
  val pred = model map { cl =>
    cl.classifyInstance(Instance('size -> 10,
      'weight -> 50))
  }
  println(pred)
```

For more basic examples, see `wela.examples.MyApp`.

contribute
----------

The project follows the [git flow](http://nvie.com/posts/a-successful-git-branching-model/) organization for branching. If you want to participate, check out how it works.

License
-------

Following WEKA's GPL licence, this project is also under the [GNU General Public Licence V3](http://www.gnu.org/licenses/gpl.html).
