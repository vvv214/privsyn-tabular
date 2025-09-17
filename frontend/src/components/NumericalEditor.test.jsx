import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import NumericalEditor from './NumericalEditor';

const buildProps = (overrides = {}) => ({
  columnKey: overrides.columnKey || 'value',
  domainDetails: {
    bounds: { min: 0, max: 10 },
    binning: { method: 'uniform', bin_count: 5, bin_width: '', growth_rate: '', dp_budget_fraction: 0.05 },
    numeric_summary: { min: 0, max: 10 },
    ...overrides.domainDetails,
  },
  onChange: overrides.onChange || vi.fn(),
  showNonNumericWarning: overrides.showNonNumericWarning ?? false,
});

describe('NumericalEditor', () => {
  afterEach(() => {
    cleanup();
  });

  it('blocks invalid numeric input and surfaces error message', () => {
    const props = buildProps();
    const { onChange } = props;
    render(<NumericalEditor {...props} />);

    const minInput = screen.getByLabelText(/minimum/i);
    Object.defineProperty(minInput, 'validity', { value: { badInput: true } });
    fireEvent.change(minInput, { target: { value: '' } });

    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByText(/enter a finite number/i)).toBeVisible();
  });

  it('allows extremely large values but shows a warning', () => {
    const onChange = vi.fn();
    const props = buildProps({ onChange });
    render(<NumericalEditor {...props} />);

    const maxInput = screen.getByLabelText(/maximum/i);
    fireEvent.change(maxInput, { target: { value: '1e12' } });

    expect(onChange).toHaveBeenCalled();
    expect(screen.getByText(/value is extremely large/i)).toBeVisible();
  });
});
