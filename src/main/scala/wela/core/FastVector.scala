package wela.core

import weka.core.{ FastVector => WekaFastVector }
import scala.collection.generic.GenericTraversableTemplate
import scala.collection.generic.TraversableFactory
import scala.collection.generic.CanBuildFrom
import scala.collection.mutable.Builder
import scala.collection.TraversableLike
import scala.collection.IndexedSeqLike

object FastVector extends TraversableFactory[FastVector] {

  implicit def canBuildFrom[E]: CanBuildFrom[Coll, E, FastVector[E]] = new GenericCanBuildFrom[E]

  def newBuilder[E] = new Builder[E, FastVector[E]] {
    private var parts: WekaFastVector = new WekaFastVector()
    override def +=(elem: E): this.type = {
      parts.addElement(elem)
      this
    }
    override def clear() = {
      parts.removeAllElements()
    }
    override def result(): FastVector[E] = {
      new FastVector(parts)
    }

    override def sizeHint(size: Int) {
      if (parts.capacity() < size) {
        val newParts = new WekaFastVector(size)
        if (parts.size() > 0) {
          newParts.appendElements(parts)
        }
        parts = newParts
      }
    }

  }

}

/**
 * a wrapper around a Weka FastVector, but with traversable like methods
 */
class FastVector[+E] protected[core] (vect: WekaFastVector) extends IndexedSeq[E]
  with GenericTraversableTemplate[E, FastVector]
  with IndexedSeqLike[E, FastVector[E]]
  with TraversableLike[E, FastVector[E]] {
  override def companion = FastVector

  private val theFastVector: WekaFastVector = vect
  protected[core] lazy val wrapped: WekaFastVector = theFastVector.copy.asInstanceOf[WekaFastVector]

  override def foreach[U](f: E => U) {
    val enum = wrapped.elements
    while (enum.hasMoreElements) {
      val cast = enum.nextElement.asInstanceOf[E]
      f(cast)
    }
  }
  
  override def apply(idx: Int): E = theFastVector.elementAt(idx).asInstanceOf[E]
  override def length: Int = theFastVector.size
  override def seq: IndexedSeq[E] = this.toIndexedSeq

}