# Elegant construction of objects referring to each other in final fields

Recently, it struck me that [Java Lambda Expressions](https://docs.oracle.com/javase/tutorial/java/javaOO/lambdaexpressions.html)
allow me to solve a problem I recently had quite elegantly.

Let's say you have two classes, `Foo` and `Bar` with the following constraints:
- `Foo` must refer to a `Bar` object and vice-versa.
- The fields storing references to the other object must be final (i.e., you
  can't construct one and then set it on the other).
- As you cannot construct both objects and then link them (because the fields are
  `final`), you could instead defer the construction of `Bar` to the `Foo` constructor
  and call `this.bar = new Bar(this)` there. However, if
  `Bar` has additional constructor arguments, you would have to pass them into 
  the `Foo` constructor as well, thus polluting it. We don't want that.
- You want to control the construction of both objects at the same place (which also
  disallows the workaround proposed in the previous bullet point).

Here's the solution using lambdas:

```java
class Foo {
    final Bar bar;
    
    Foo(Function<Foo, Bar> barCreator, int fooExtraArgs) {
        this.bar = barCreator.apply(this);
    }
}

class Bar {
    final Foo foo;
    
    Bar(Foo foo, int barExtraArgs) {
        this.foo = foo;
    }
}

public class LinkedObjects {
    public static void main(String[] args) {
        // controlling 'Bar' construction
        int barExtraArgs = 42;
        Function<Foo, Bar> barCreator = foo -> new Bar(foo, barExtraArgs);
        
        // controlling 'Foo' construction
        int fooExtraArgs = 666;
        Foo foo = new Foo(barCreator, fooExtraArgs);
    }
}
```
