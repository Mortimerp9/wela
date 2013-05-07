package wela

import weka.core.{ FastVector => WekaFastVector, Attribute => WekaAttribute, Instance => WekaInstance }
package object core {

  implicit def fastVectorToWela[E](fv: WekaFastVector): FastVector[E] = new FastVector(fv)
  implicit def welaToFastVector[E](fv: FastVector[E]): WekaFastVector = fv.wrapped

  implicit def welaAttrToWeka(attr: Attribute): WekaAttribute = attr.toWekaAttribute

  type AttributeSet = Map[Symbol, Attribute]
  implicit def attrSet(attrs: Seq[Attribute]): AttributeSet = {
    val m = attrs.map { a =>
      a.name -> a
    }
    m.toMap
  }

  type Instance = FastVector[(Symbol, AttributeValue)]
  object Instance {
    def apply(vals: (Symbol, AttributeValue)*): Instance = FastVector(vals: _*)
  }

  implicit def avToVal(av: AttributeValue): av.T = av.value
  implicit def strToAV(string: String) = StringValue(string)
  implicit def strToAV(string: Symbol) = StringValue(string.name)
  implicit def dblToAV(double: Double) = DoubleValue(double)

}


