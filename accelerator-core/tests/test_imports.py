"""
Test that all imports work correctly in the accelerator-core package.
"""

def test_core_imports():
    """Test that core components can be imported."""
    try:
        from accelerator.core.runtime.context import Context
        from accelerator.core.hooks import HookRegistry
        from accelerator.core.typings import ConfigType, MetricsDict
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_namespace_structure():
    """Test that the namespace package structure is correct."""
    import accelerator.core
    assert hasattr(accelerator.core, '__path__')

if __name__ == "__main__":
    test_core_imports()
    test_namespace_structure()
    print("All import tests passed!")