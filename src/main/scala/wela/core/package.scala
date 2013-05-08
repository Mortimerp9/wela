package wela

import weka.core.{ FastVector => WekaFastVector, Attribute => WekaAttribute, Instance => WekaInstance }
package object core {

  type Tagged[U] = { type Tag = U }
  type @@[T, U] = T with Tagged[U]

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

  type Instance = Map[Symbol, AttributeValue]
  object Instance {
    def apply(vals: (Symbol, AttributeValue)*): Instance = Map(vals: _*)
  }

  implicit def strToAV(string: String): StringValue = string.asInstanceOf[StringValue] 
  implicit def strToAV(string: Symbol): SymbolValue = string.asInstanceOf[SymbolValue]
  implicit def dblToAV(double: Double): NumericValue = double.asInstanceOf[NumericValue]

  trait AttributeValueTag
  type AttributeValue = Tagged[AttributeValueTag]
  type StringValue = String @@ AttributeValueTag
  type SymbolValue = Symbol @@ AttributeValueTag
  type NumericValue = Double @@ AttributeValueTag
  
  trait Compatible[AD <:Attribute, AV <:AttributeValue]
  implicit val compString = new Compatible[StringAttribute, StringValue]{}
  implicit val compSymbol = new Compatible[NominalAttribute, SymbolValue]{}
  implicit val compNumeric = new Compatible[NumericAttribute, NumericValue]{}
  
}


