"""Unit tests for ContainerUpdateTransform and related classes."""

import pytest

from accelerator.core.transform.container_transform import (
    ContainerCondition,
    ContainerPrinter,
    ContainerUpdateTransform,
)


class TestContainerUpdateTransformInit:
    """Tests for ContainerUpdateTransform.__init__ method."""
    
    def test_init_minimal_config(self):
        """Test initialization with minimal configuration."""
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb']
        )
        
        assert transform.items == 'per_step.prediction.output'
        assert transform.input_getter is not None
        assert transform.output_setter is not None
        assert transform.transform is None  # No trans_opt provided
        assert transform.condition is None
        assert transform.print_in is None
        assert transform.print_out is None
        assert transform.copy_context is False
    
    def test_init_with_trans_opt(self):
        """Test initialization with transform options."""
        trans_opt = {
            'type': 'DummyTransform',
            'param1': 'value1',
            'param2': 42
        }
        
        # DummyTransform doesn't exist in registry, so this should raise KeyError
        with pytest.raises(KeyError, match="Transform 'DummyTransform' not found"):
            ContainerUpdateTransform(
                items='per_step.prediction.output',
                trans_inputs=['per_batch.input.rgb'],
                trans_opt=trans_opt
            )
    
    def test_init_trans_inputs_as_list(self):
        """Test initialization with trans_inputs as list (positional args)."""
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb', 'per_batch.target.label']
        )
        
        assert transform.input_getter.args_paths == [
            'per_batch.input.rgb',
            'per_batch.target.label'
        ]
        assert transform.input_getter.kwargs_paths == {}
    
    def test_init_trans_inputs_as_dict_with_args_kwargs(self):
        """Test initialization with trans_inputs as dict with args/kwargs keys."""
        trans_inputs = {
            'args': ['per_batch.input.rgb'],
            'kwargs': {'pca_mat': 'persistent.additional.pca_mat'}
        }
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=trans_inputs
        )
        
        assert transform.input_getter.args_paths == ['per_batch.input.rgb']
        assert transform.input_getter.kwargs_paths == {'pca_mat': 'persistent.additional.pca_mat'}
    
    def test_init_trans_inputs_as_dict_kwargs_only(self):
        """Test initialization with trans_inputs as dict (backward compatibility)."""
        trans_inputs = {
            'input': 'per_batch.input.rgb',
            'target': 'per_batch.target.label'
        }
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=trans_inputs
        )
        
        assert transform.input_getter.args_paths == []
        assert transform.input_getter.kwargs_paths == trans_inputs
    
    def test_init_trans_inputs_none(self):
        """Test initialization with no trans_inputs (defaults to empty list)."""
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=None
        )
        
        assert transform.input_getter.args_paths == []
        assert transform.input_getter.kwargs_paths == {}
    
    def test_init_multiple_output_items(self):
        """Test initialization with multiple output paths."""
        items = ['per_step.loss.ce', 'per_step.loss.reg']
        
        transform = ContainerUpdateTransform(
            items=items,
            trans_inputs=['per_batch.prediction.output', 'per_batch.target.label']
        )
        
        assert transform.items == items
        assert transform.output_setter.items == items
    
    def test_init_with_condition(self):
        """Test initialization with condition configuration."""
        condition = {
            'path': 'persistent.config.enabled',
            'operator': 'equal',
            'value': True
        }
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            condition=condition
        )
        
        assert transform.condition is not None
        assert isinstance(transform.condition, ContainerCondition)
    
    def test_init_with_print_in_bool(self):
        """Test initialization with print_in as boolean."""
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            print_in=True
        )
        
        assert transform.print_in is not None
        assert isinstance(transform.print_in, ContainerPrinter)
    
    def test_init_with_print_out_dict(self):
        """Test initialization with print_out as dict."""
        print_config = {'banner': True, 'max_items': 10}
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            print_out=print_config
        )
        
        assert transform.print_out is not None
        assert isinstance(transform.print_out, ContainerPrinter)
    
    def test_init_with_copy_context(self):
        """Test initialization with copy_context flag."""
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            copy_context=True
        )
        
        assert transform.copy_context is True
    
    def test_init_without_items(self):
        """Test initialization without output items (passthrough mode)."""
        transform = ContainerUpdateTransform(
            trans_inputs=['per_batch.input.rgb']
        )
        
        assert transform.items is None
        assert transform.output_setter is None
    
    def test_init_trans_opt_missing_type(self):
        """Test that initialization fails if trans_opt missing 'type' key."""
        with pytest.raises(ValueError, match="trans_opt must include 'type' key"):
            ContainerUpdateTransform(
                items='per_step.prediction.output',
                trans_inputs=['per_batch.input.rgb'],
                trans_opt={'param1': 'value1'}  # Missing 'type'
            )
    
    def test_init_trans_inputs_invalid_type(self):
        """Test that initialization fails with invalid trans_inputs type."""
        with pytest.raises(ValueError, match="trans_inputs must be a list or dict"):
            ContainerUpdateTransform(
                items='per_step.prediction.output',
                trans_inputs="invalid_string"  # Should be list or dict
            )
    
    def test_repr(self):
        """Test string representation of ContainerUpdateTransform."""
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            condition={'path': 'test', 'operator': 'equal', 'value': True}
        )
        
        repr_str = repr(transform)
        assert 'ContainerUpdateTransform' in repr_str
        assert 'per_step.prediction.output' in repr_str
        assert 'has_condition=True' in repr_str


class TestContainerCondition:
    """Tests for ContainerCondition class."""
    
    def test_init_with_none(self):
        """Test initialization with None config."""
        condition = ContainerCondition(None)
        assert condition.condition_config == {}
    
    def test_init_with_config(self):
        """Test initialization with condition config."""
        config = {'path': 'test.path', 'operator': 'equal', 'value': 42}
        condition = ContainerCondition(config)
        assert condition.condition_config == config
    
    def test_init_missing_operator(self):
        """Test initialization fails with missing operator."""
        with pytest.raises(ValueError, match="must specify 'operator'"):
            ContainerCondition({'path': 'test.path', 'value': 42})
    
    def test_init_missing_value(self):
        """Test initialization fails with missing value."""
        with pytest.raises(ValueError, match="must specify 'value'"):
            ContainerCondition({'path': 'test.path', 'operator': 'equal'})
    
    def test_call_empty_config(self):
        """Test that empty config returns True."""
        condition = ContainerCondition({})
        assert condition(None) is True
    
    def test_call_equal_operator_true(self):
        """Test equal operator when condition is met."""
        context = MockContext()
        context._data['persistent.config.enabled'] = True
        
        condition = ContainerCondition({
            'path': 'persistent.config.enabled',
            'operator': 'equal',
            'value': True
        })
        
        assert condition(context) is True
    
    def test_call_equal_operator_false(self):
        """Test equal operator when condition is not met."""
        context = MockContext()
        context._data['persistent.config.enabled'] = False
        
        condition = ContainerCondition({
            'path': 'persistent.config.enabled',
            'operator': 'equal',
            'value': True
        })
        
        assert condition(context) is False
    
    def test_call_not_equal_operator(self):
        """Test not_equal operator."""
        context = MockContext()
        context._data['persistent.config.mode'] = 'train'
        
        condition = ContainerCondition({
            'path': 'persistent.config.mode',
            'operator': 'not_equal',
            'value': 'test'
        })
        
        assert condition(context) is True
    
    def test_call_greater_than_operator(self):
        """Test greater_than operator."""
        context = MockContext()
        context._data['per_epoch.metrics.accuracy'] = 0.95
        
        condition = ContainerCondition({
            'path': 'per_epoch.metrics.accuracy',
            'operator': 'greater_than',
            'value': 0.9
        })
        
        assert condition(context) is True
    
    def test_call_less_than_operator(self):
        """Test less_than operator."""
        context = MockContext()
        context._data['per_step.loss.total'] = 0.1
        
        condition = ContainerCondition({
            'path': 'per_step.loss.total',
            'operator': 'less_than',
            'value': 0.5
        })
        
        assert condition(context) is True
    
    def test_call_greater_equal_operator(self):
        """Test greater_equal operator."""
        context = MockContext()
        context._data['per_epoch.metrics.accuracy'] = 0.9
        
        condition = ContainerCondition({
            'path': 'per_epoch.metrics.accuracy',
            'operator': 'greater_equal',
            'value': 0.9
        })
        
        assert condition(context) is True
    
    def test_call_less_equal_operator(self):
        """Test less_equal operator."""
        context = MockContext()
        context._data['per_step.loss.total'] = 0.5
        
        condition = ContainerCondition({
            'path': 'per_step.loss.total',
            'operator': 'less_equal',
            'value': 0.5
        })
        
        assert condition(context) is True
    
    def test_call_in_operator(self):
        """Test in operator."""
        context = MockContext()
        context._data['persistent.config.mode'] = 'train'
        
        condition = ContainerCondition({
            'path': 'persistent.config.mode',
            'operator': 'in',
            'value': ['train', 'finetune', 'eval']
        })
        
        assert condition(context) is True
    
    def test_call_not_in_operator(self):
        """Test not_in operator."""
        context = MockContext()
        context._data['persistent.config.mode'] = 'test'
        
        condition = ContainerCondition({
            'path': 'persistent.config.mode',
            'operator': 'not_in',
            'value': ['train', 'finetune']
        })
        
        assert condition(context) is True
    
    def test_call_multiple_conditions_and_all_true(self):
        """Test multiple conditions with AND logic (all true)."""
        context = MockContext()
        context._data['persistent.config.enabled'] = True
        context._data['per_epoch.metrics.accuracy'] = 0.95
        
        condition = ContainerCondition({
            'logic': 'AND',
            'conditions': [
                {'path': 'persistent.config.enabled', 'operator': 'equal', 'value': True},
                {'path': 'per_epoch.metrics.accuracy', 'operator': 'greater_than', 'value': 0.9}
            ]
        })
        
        assert condition(context) is True
    
    def test_call_multiple_conditions_and_one_false(self):
        """Test multiple conditions with AND logic (one false)."""
        context = MockContext()
        context._data['persistent.config.enabled'] = True
        context._data['per_epoch.metrics.accuracy'] = 0.85
        
        condition = ContainerCondition({
            'logic': 'AND',
            'conditions': [
                {'path': 'persistent.config.enabled', 'operator': 'equal', 'value': True},
                {'path': 'per_epoch.metrics.accuracy', 'operator': 'greater_than', 'value': 0.9}
            ]
        })
        
        assert condition(context) is False
    
    def test_call_multiple_conditions_or_one_true(self):
        """Test multiple conditions with OR logic (one true)."""
        context = MockContext()
        context._data['persistent.config.mode'] = 'train'
        
        condition = ContainerCondition({
            'logic': 'OR',
            'conditions': [
                {'path': 'persistent.config.mode', 'operator': 'equal', 'value': 'train'},
                {'path': 'persistent.config.mode', 'operator': 'equal', 'value': 'finetune'}
            ]
        })
        
        assert condition(context) is True
    
    def test_call_multiple_conditions_or_all_false(self):
        """Test multiple conditions with OR logic (all false)."""
        context = MockContext()
        context._data['persistent.config.mode'] = 'test'
        
        condition = ContainerCondition({
            'logic': 'OR',
            'conditions': [
                {'path': 'persistent.config.mode', 'operator': 'equal', 'value': 'train'},
                {'path': 'persistent.config.mode', 'operator': 'equal', 'value': 'finetune'}
            ]
        })
        
        assert condition(context) is False
    
    def test_call_multiple_conditions_default_and(self):
        """Test multiple conditions default to AND logic."""
        context = MockContext()
        context._data['persistent.config.enabled'] = True
        context._data['per_epoch.metrics.accuracy'] = 0.95
        
        condition = ContainerCondition({
            'conditions': [
                {'path': 'persistent.config.enabled', 'operator': 'equal', 'value': True},
                {'path': 'per_epoch.metrics.accuracy', 'operator': 'greater_than', 'value': 0.9}
            ]
        })
        
        assert condition(context) is True
    
    def test_call_path_not_found(self):
        """Test condition when path is not found in context."""
        context = MockContext()
        # Make get_item raise an error
        context.get_item = lambda path: (_ for _ in ()).throw(ValueError(f"Path not found: {path}"))
        
        condition = ContainerCondition({
            'path': 'nonexistent.path',
            'operator': 'equal',
            'value': True
        })
        
        # Should return False when path not found
        assert condition(context) is False
    
    def test_call_unknown_operator(self):
        """Test condition with unknown operator."""
        context = MockContext()
        context._data['persistent.config.enabled'] = True
        
        condition = ContainerCondition({
            'path': 'persistent.config.enabled',
            'operator': 'unknown_operator',
            'value': True
        })
        
        # Should return False for unknown operator
        assert condition(context) is False
    
    def test_repr(self):
        """Test string representation of ContainerCondition."""
        config = {'path': 'test.path', 'operator': 'equal', 'value': 42}
        condition = ContainerCondition(config)
        
        repr_str = repr(condition)
        assert 'ContainerCondition' in repr_str
        assert 'test.path' in repr_str


class TestContainerPrinter:
    """Tests for ContainerPrinter class."""
    
    def test_init_with_bool(self):
        """Test initialization with boolean config."""
        printer = ContainerPrinter('IN', True)
        assert printer.label == 'IN'
        assert printer.print_config == {}
        assert printer.banner is True
    
    def test_init_with_dict(self):
        """Test initialization with dict config."""
        config = {'banner': True, 'max_items': 10, 'banner_char': '-'}
        printer = ContainerPrinter('OUT', config)
        assert printer.label == 'OUT'
        assert printer.print_config == config
        assert printer.max_items == 10
        assert printer.banner_char == '-'
    
    def test_init_defaults(self):
        """Test that defaults are set correctly."""
        printer = ContainerPrinter('IN', {})
        assert printer.banner is True
        assert printer.banner_char == '='
        assert printer.banner_width == 80
        assert printer.max_items == 10
        assert printer.max_str_len == 100
        assert printer.indent == 2
    
    def test_call_with_simple_context(self, capsys):
        """Test printing with simple context."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 42
        
        printer = ContainerPrinter('IN', {'paths': ['per_batch.input.rgb']})
        printer(context)
        
        captured = capsys.readouterr()
        assert 'Transform IN' in captured.out
        assert 'per_batch.input.rgb: 42' in captured.out
    
    def test_call_with_banner(self, capsys):
        """Test that banner is printed."""
        context = MockContext()
        
        printer = ContainerPrinter('OUT', {'banner': True, 'banner_char': '='})
        printer(context)
        
        captured = capsys.readouterr()
        assert '=' in captured.out
        assert 'Transform OUT' in captured.out
    
    def test_call_without_banner(self, capsys):
        """Test printing without banner."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 42
        
        printer = ContainerPrinter('IN', {
            'paths': ['per_batch.input.rgb'],
            'banner': False
        })
        printer(context)
        
        captured = capsys.readouterr()
        assert 'Transform IN' not in captured.out
        assert 'per_batch.input.rgb: 42' in captured.out
    
    def test_call_with_multiple_paths(self, capsys):
        """Test printing multiple paths."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        context._data['per_step.loss.total'] = 0.5
        
        printer = ContainerPrinter('OUT', {
            'paths': ['per_batch.input.rgb', 'per_step.loss.total'],
            'banner': False
        })
        printer(context)
        
        captured = capsys.readouterr()
        assert 'per_batch.input.rgb: 10' in captured.out
        assert 'per_step.loss.total: 0.5' in captured.out
    
    def test_call_with_missing_path(self, capsys):
        """Test printing when path doesn't exist."""
        context = MockContext()
        context.get_item = lambda path: (_ for _ in ()).throw(ValueError(f"Path not found: {path}"))
        
        printer = ContainerPrinter('IN', {
            'paths': ['nonexistent.path'],
            'banner': False
        })
        printer(context)
        
        captured = capsys.readouterr()
        assert 'nonexistent.path: <Error:' in captured.out
    
    def test_call_without_paths_prints_summary(self, capsys):
        """Test that summary is printed when no paths specified."""
        context = MockContext()
        
        printer = ContainerPrinter('IN', {'banner': False})
        printer(context)
        
        captured = capsys.readouterr()
        assert 'Context Summary' in captured.out
        assert 'Type: MockContext' in captured.out
    
    def test_format_value_none(self):
        """Test formatting None value."""
        printer = ContainerPrinter('IN', {})
        assert printer._format_value(None) == 'None'
    
    def test_format_value_basic_types(self):
        """Test formatting basic types."""
        printer = ContainerPrinter('IN', {})
        assert printer._format_value(42) == '42'
        assert printer._format_value(3.14) == '3.14'
        assert printer._format_value(True) == 'True'
    
    def test_format_value_string(self):
        """Test formatting strings."""
        printer = ContainerPrinter('IN', {})
        assert printer._format_value('hello') == "'hello'"
    
    def test_format_value_long_string(self):
        """Test formatting long strings (truncation)."""
        printer = ContainerPrinter('IN', {'max_str_len': 10})
        long_str = 'a' * 50
        result = printer._format_value(long_str)
        assert 'truncated' in result
        assert 'length=50' in result
    
    def test_format_value_list(self):
        """Test formatting lists."""
        printer = ContainerPrinter('IN', {})
        assert printer._format_value([]) == 'list(empty)'
        assert printer._format_value([1, 2, 3]) == 'list([1, 2, 3])'
    
    def test_format_value_long_list(self):
        """Test formatting long lists (truncation)."""
        printer = ContainerPrinter('IN', {'max_items': 3})
        long_list = list(range(20))
        result = printer._format_value(long_list)
        assert 'list([0, 1, 2, ...' in result
        assert '20 items total' in result
    
    def test_format_value_dict(self):
        """Test formatting dictionaries."""
        printer = ContainerPrinter('IN', {})
        assert printer._format_value({}) == 'dict(empty)'
        result = printer._format_value({'a': 1, 'b': 2})
        assert 'dict({' in result
        assert 'a: 1' in result
        assert 'b: 2' in result
    
    def test_format_value_long_dict(self):
        """Test formatting long dictionaries (truncation)."""
        printer = ContainerPrinter('IN', {'max_items': 2})
        long_dict = {f'key{i}': i for i in range(10)}
        result = printer._format_value(long_dict)
        assert '10 items total' in result
    
    def test_repr(self):
        """Test string representation of ContainerPrinter."""
        config = {'banner': True, 'max_items': 5}
        printer = ContainerPrinter('IN', config)
        
        repr_str = repr(printer)
        assert 'ContainerPrinter' in repr_str
        assert "label='IN'" in repr_str



class MockContext:
    """Mock context for testing."""
    
    def __init__(self):
        self._data = {}
    
    def get_item(self, path: str):
        """Mock get_item."""
        return self._data.get(path)
    
    def set_item(self, path: str, value, use_weakref=None):
        """Mock set_item."""
        self._data[path] = value


class MockTransform:
    """Mock transform for testing."""
    
    def __init__(self, multiplier=2):
        self.multiplier = multiplier
    
    def __call__(self, *args, **kwargs):
        """Mock transform that multiplies first arg by multiplier."""
        if args:
            return args[0] * self.multiplier
        return None


class TestContainerUpdateTransformCall:
    """Tests for ContainerUpdateTransform.__call__ method."""
    
    def test_call_simple_passthrough(self):
        """Test __call__ with simple passthrough (no transform)."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 42
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb']
        )
        
        result_context = transform(context)
        
        assert result_context._data['per_step.prediction.output'] == 42
    
    def test_call_with_transform(self):
        """Test __call__ with actual transform applied."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        
        # Create transform with mock transform object
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb']
        )
        transform.transform = MockTransform(multiplier=3)
        
        result_context = transform(context)
        
        assert result_context._data['per_step.prediction.output'] == 30
    
    def test_call_multiple_inputs_outputs(self):
        """Test __call__ with multiple inputs and outputs."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = [1, 2, 3]
        context._data['per_batch.target.label'] = [4, 5, 6]
        
        # Mock transform that returns tuple
        transform = ContainerUpdateTransform(
            items=['per_step.loss.ce', 'per_step.loss.reg'],
            trans_inputs=['per_batch.input.rgb', 'per_batch.target.label']
        )
        transform.transform = lambda x, y: (sum(x), sum(y))
        
        result_context = transform(context)
        
        assert result_context._data['per_step.loss.ce'] == 6  # sum([1,2,3])
        assert result_context._data['per_step.loss.reg'] == 15  # sum([4,5,6])
    
    def test_call_with_kwargs(self):
        """Test __call__ with keyword arguments."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 5
        context._data['persistent.additional.multiplier'] = 7
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs={
                'kwargs': {
                    'value': 'per_batch.input.rgb',
                    'mult': 'persistent.additional.multiplier'
                }
            }
        )
        transform.transform = lambda value, mult: value * mult
        
        result_context = transform(context)
        
        assert result_context._data['per_step.prediction.output'] == 35
    
    def test_call_with_condition_met(self):
        """Test __call__ when condition is met."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        context._data['persistent.config.enabled'] = True
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            condition={'path': 'persistent.config.enabled', 'operator': 'equal', 'value': True}
        )
        transform.transform = MockTransform(multiplier=2)
        
        result_context = transform(context)
        
        # Transform should execute
        assert result_context._data['per_step.prediction.output'] == 20
    
    def test_call_with_condition_not_met(self):
        """Test __call__ when condition is not met."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            condition={'path': 'persistent.config.enabled', 'operator': 'equal', 'value': True}
        )
        transform.transform = MockTransform(multiplier=2)
        
        # Override condition to return False
        transform.condition = lambda ctx: False
        
        result_context = transform(context)
        
        # Transform should NOT execute, output should not be set
        assert 'per_step.prediction.output' not in result_context._data
    
    def test_call_with_copy_context(self):
        """Test __call__ with copy_context=True."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            copy_context=True
        )
        
        result_context = transform(context)
        
        # Result should be a different object
        assert result_context is not context
        # But should have the output set
        assert result_context._data['per_step.prediction.output'] == 10
    
    def test_call_without_output_setter(self):
        """Test __call__ without output items (no setter)."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        
        transform = ContainerUpdateTransform(
            trans_inputs=['per_batch.input.rgb']
        )
        transform.transform = MockTransform(multiplier=2)
        
        result_context = transform(context)
        
        # Should execute but not set any outputs
        assert result_context is context
        assert 'per_step.prediction.output' not in result_context._data
    
    def test_call_transform_execution_error(self):
        """Test __call__ when transform raises an error."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb']
        )
        transform.transform = lambda x: 1 / 0  # Will raise ZeroDivisionError
        
        with pytest.raises(RuntimeError, match="Transform execution failed"):
            transform(context)
    
    def test_call_input_extraction_error(self):
        """Test __call__ when input extraction fails."""
        context = MockContext()
        # Make get_item raise an error
        context.get_item = lambda path: (_ for _ in ()).throw(ValueError(f"Path not found: {path}"))
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb']
        )
        
        with pytest.raises(RuntimeError, match="Failed to extract inputs"):
            transform(context)
    
    def test_call_output_setting_error(self):
        """Test __call__ when output setting fails."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        
        # Create transform that returns wrong number of outputs
        transform = ContainerUpdateTransform(
            items=['per_step.loss.ce', 'per_step.loss.reg'],  # Expects 2 outputs
            trans_inputs=['per_batch.input.rgb']
        )
        transform.transform = lambda x: x  # Returns only 1 value
        
        with pytest.raises(RuntimeError, match="Failed to set outputs"):
            transform(context)
    
    def test_call_with_print_in_and_out(self):
        """Test __call__ with print_in and print_out enabled."""
        context = MockContext()
        context._data['per_batch.input.rgb'] = 10
        
        transform = ContainerUpdateTransform(
            items='per_step.prediction.output',
            trans_inputs=['per_batch.input.rgb'],
            print_in=True,
            print_out=True
        )
        
        # Should not raise even with printers enabled
        result_context = transform(context)
        assert result_context._data['per_step.prediction.output'] == 10
