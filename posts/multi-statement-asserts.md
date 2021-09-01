# Asserting the result of a multi-statement computation

So you want to assert something in your Java code, but that "something" requires
a few lines of code to compute? There are two solutions to this problem.

## Put the code to assert into a method

```java
class MultiStatementAsserts1 {
    public void code() {
        // code prior to the assert
        // ...

        assert someComplexCondition();
    }

    private boolean someComplexCondition() {
        // some code that evaluates to a boolean result
        // ...
        // ...
        return result;
    }
}
```

That works, but goes against the philosophy that asserts should be non-invasive, and
adding a dedicated method just for an assert is quite invasive. Can we do better? Yes, with `BooleanSupplier` (which, if required, can capture local
variables and/or state):

## Asserting the result of a boolean supplier

```java
class MultiStatementAsserts2 {
    public void code() {
        // code prior to the assert
        // ...

        assert ((BooleanSupplier) () -> {
            // some code that evaluates to a boolean result
            // ...
            // ...
            return result;
        }).getAsBoolean();
    }
}
```
