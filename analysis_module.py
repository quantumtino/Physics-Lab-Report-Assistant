import numpy as np
from scipy import fft, optimize
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Optional

class DataAnalyzer:
    def __init__(self):
        pass

    def format_with_uncertainty(self, value: float, uncertainty: float, sig: int = 2) -> Tuple[str, str]:
        """
        格式化物理量及不确定度：不确定度保留2位有效数字，数值与之对齐小数位
        返回 (value_str, unc_str)
        """
        if uncertainty is None or uncertainty <= 0 or math.isnan(uncertainty):
            return f"{value:.4g}", "0"

        exp = math.floor(math.log10(abs(uncertainty)))
        dec = max(0, sig - 1 - exp)
        unc_rounded = round(uncertainty, dec)
        val_rounded = round(value, dec)

        fmt = f"{{:.{dec}f}}"
        return fmt.format(val_rounded), fmt.format(unc_rounded)
    
    def linear_fit(self, x_data: List[float], y_data: List[float],
                   y_err: Optional[List[float]] = None,
                   x_err: Optional[List[float]] = None) -> Tuple[float, float, float, float, float, float]:
        """
        线性拟合 y = ax + b，并返回参数不确定度；当提供 y_err 时使用加权最小二乘。
        :param x_data: x轴数据
        :param y_data: y轴数据
        :param y_err: y 方向标准不确定度（与误差棒一致）。
        :param x_err: x 不确定度（目前未用于拟合，仅用于将来扩展 York 方法）。
        :return: (斜率a, 截距b, 加权R², a的不确定度, b的不确定度, 减少卡方)
        """
        x = np.array(x_data)
        y = np.array(y_data)

        use_weights = y_err is not None and len(y_err) == len(y) and np.all(np.array(y_err) > 0)

        if use_weights:
            w = 1.0 / (np.array(y_err) ** 2)
            S = w.sum()
            Sx = (w * x).sum()
            Sy = (w * y).sum()
            Sxx = (w * x * x).sum()
            Sxy = (w * x * y).sum()
            Delta = S * Sxx - Sx * Sx
            slope = (S * Sxy - Sx * Sy) / Delta
            intercept = (Sy - slope * Sx) / S

            # 残差与加权R²
            y_hat = slope * x + intercept
            y_bar_w = Sy / S
            SSE = (w * (y - y_hat) ** 2).sum()
            SST = (w * (y - y_bar_w) ** 2).sum()
            r_squared_w = 1.0 - (SSE / SST if SST > 0 else 0.0)

            dof = max(1, len(x) - 2)
            chi2_red = SSE / dof
            var_slope = S / Delta
            var_intercept = Sxx / Delta
            slope_err = float(np.sqrt(var_slope * chi2_red))
            intercept_err = float(np.sqrt(var_intercept * chi2_red))

            return float(slope), float(intercept), float(r_squared_w), slope_err, intercept_err, float(chi2_red)
        else:
            # 非加权：退回到 polyfit 协方差
            (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
            slope_err = float(np.sqrt(cov[0, 0])) if cov.size else 0.0
            intercept_err = float(np.sqrt(cov[1, 1])) if cov.size else 0.0
            correlation_matrix = np.corrcoef(x, y)
            r_squared = float(correlation_matrix[0, 1] ** 2)
            # 估计 reduced chi^2
            y_hat = slope * x + intercept
            resid = y - y_hat
            dof = max(1, len(x) - 2)
            chi2_red = float((resid ** 2).sum() / dof)
            return float(slope), float(intercept), float(r_squared), slope_err, intercept_err, chi2_red

    def log_fit(self, x_data: List[float], y_data: List[float],
                y_err: Optional[List[float]] = None) -> Tuple[float, float, float, float, float, float]:
        """
        对 y = a * ln(x) + b 拟合，并返回参数不确定度（x 必须 > 0）；提供 y_err 时使用加权最小二乘。
        :param x_data: x轴数据（需全部 > 0）
        :param y_data: y轴数据
        :param y_err: y 方向标准不确定度
        :return: (系数a, 截距b, 加权R², a的不确定度, b的不确定度, 减少卡方)
        """
        x = np.array(x_data)
        y = np.array(y_data)

        if np.any(x <= 0):
            raise ValueError("Log fit requires all x values to be > 0")

        x_log = np.log(x)
        use_weights = y_err is not None and len(y_err) == len(y) and np.all(np.array(y_err) > 0)

        if use_weights:
            w = 1.0 / (np.array(y_err) ** 2)
            S = w.sum()
            Sx = (w * x_log).sum()
            Sy = (w * y).sum()
            Sxx = (w * x_log * x_log).sum()
            Sxy = (w * x_log * y).sum()
            Delta = S * Sxx - Sx * Sx
            slope = (S * Sxy - Sx * Sy) / Delta
            intercept = (Sy - slope * Sx) / S

            y_hat = slope * x_log + intercept
            y_bar_w = Sy / S
            SSE = (w * (y - y_hat) ** 2).sum()
            SST = (w * (y - y_bar_w) ** 2).sum()
            r_squared_w = 1.0 - (SSE / SST if SST > 0 else 0.0)

            dof = max(1, len(x_log) - 2)
            chi2_red = SSE / dof
            var_slope = S / Delta
            var_intercept = Sxx / Delta
            slope_err = float(np.sqrt(var_slope * chi2_red))
            intercept_err = float(np.sqrt(var_intercept * chi2_red))
            return float(slope), float(intercept), float(r_squared_w), slope_err, intercept_err, float(chi2_red)
        else:
            (slope, intercept), cov = np.polyfit(x_log, y, 1, cov=True)
            slope_err = float(np.sqrt(cov[0, 0])) if cov.size else 0.0
            intercept_err = float(np.sqrt(cov[1, 1])) if cov.size else 0.0
            correlation_matrix = np.corrcoef(x_log, y)
            r_squared = float(correlation_matrix[0, 1] ** 2)
            y_hat = slope * x_log + intercept
            resid = y - y_hat
            dof = max(1, len(x_log) - 2)
            chi2_red = float((resid ** 2).sum() / dof)
            return float(slope), float(intercept), float(r_squared), slope_err, intercept_err, chi2_red
    
    def fourier_transform(self, data: List[float], sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        对数据进行傅里叶变换
        :param data: 输入数据
        :param sampling_rate: 采样率 (默认为1.0)
        :return: (频率数组, 幅度谱)
        """
        # 转换为numpy数组
        signal = np.array(data)
        
        # 执行快速傅里叶变换
        fft_result = fft.fft(signal)
        
        # 计算频率轴
        n = len(signal)
        freq = fft.fftfreq(n, d=1/sampling_rate)
        
        # 只取正频率部分
        positive_freq_idx = freq >= 0
        freq = freq[positive_freq_idx]
        magnitude = np.abs(fft_result[positive_freq_idx])
        
        return freq, magnitude
    
    def plot_linear_fit(self, x_data: List[float], y_data: List[float], 
                       title: str = "Linear Fit", save_path: Optional[str] = None,
                       xlabel: str = "X", ylabel: str = "Y",
                       x_err: Optional[List[float]] = None,
                       y_err: Optional[List[float]] = None,
                       slope: Optional[float] = None,
                       intercept: Optional[float] = None,
                       r_squared: Optional[float] = None,
                       slope_err: Optional[float] = None,
                       intercept_err: Optional[float] = None) -> str:
        """
        绘制线性拟合结果
        :param x_data: x轴数据
        :param y_data: y轴数据
        :param title: 图表标题
        :param save_path: 保存路径 (可选)
        :param x_err: x误差 (可选)
        :param y_err: y误差 (可选)
        :return: 图表数据 (base64编码)
        """
        # 如果未提供拟合结果，内部计算
        if slope is None or intercept is None or r_squared is None:
            slope, intercept, r_squared, slope_err, intercept_err = self.linear_fit(x_data, y_data)
        else:
            # 确保不为 None
            slope_err = slope_err if slope_err is not None else 0.0
            intercept_err = intercept_err if intercept_err is not None else 0.0
        
        # 生成拟合直线
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = slope * x_fit + intercept
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        if x_err is not None or y_err is not None:
            plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', label='Data points', color='blue', ecolor='gray', capsize=3)
        else:
            plt.scatter(x_data, y_data, label='Data points', color='blue')
        val_m, val_u = self.format_with_uncertainty(slope, slope_err if slope_err is not None else 0)
        int_m, int_u = self.format_with_uncertainty(intercept, intercept_err if intercept_err is not None else 0)
        label_text = f"Fit: y=({val_m}±{val_u})x+({int_m}±{int_u}), R²={r_squared:.3f}"
        plt.plot(x_fit, y_fit, label=label_text, color='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        # 返回图表数据
        import io
        import base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str
    
    def plot_fourier_transform(self, data: List[float], sampling_rate: float = 1.0,
                              title: str = "Fourier Transform", save_path: Optional[str] = None) -> str:
        """
        绘制傅里叶变换结果
        :param data: 输入数据
        :param sampling_rate: 采样率
        :param title: 图表标题
        :param save_path: 保存路径 (可选)
        :return: 图表数据 (base64编码)
        """
        # 执行傅里叶变换
        freq, magnitude = self.fourier_transform(data, sampling_rate)
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(freq, magnitude)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        # 返回图表数据
        import io
        import base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_str

    def plot_log_fit(self, x_data: List[float], y_data: List[float],
                    title: str = "Log Fit", save_path: Optional[str] = None,
                    xlabel: str = "X", ylabel: str = "Y",
                    x_err: Optional[List[float]] = None,
                    y_err: Optional[List[float]] = None,
                    slope: Optional[float] = None,
                    intercept: Optional[float] = None,
                    r_squared: Optional[float] = None,
                    slope_err: Optional[float] = None,
                    intercept_err: Optional[float] = None) -> str:
        """
        绘制对数拟合结果 y = a ln(x) + b
        :param x_data: x轴数据（需全部 > 0）
        :param y_data: y轴数据
        :param title: 图表标题
        :param save_path: 保存路径 (可选)
        :param x_err: x误差 (可选)
        :param y_err: y误差 (可选)
        :return: 图表数据 (base64编码)
        """
        if slope is None or intercept is None or r_squared is None:
            slope, intercept, r_squared, slope_err, intercept_err = self.log_fit(x_data, y_data)
        else:
            slope_err = slope_err if slope_err is not None else 0.0
            intercept_err = intercept_err if intercept_err is not None else 0.0

        x = np.array(x_data)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * np.log(x_fit) + intercept

        plt.figure(figsize=(10, 6))
        if x_err is not None or y_err is not None:
            plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', label='Data points', color='blue', ecolor='gray', capsize=3)
        else:
            plt.scatter(x_data, y_data, label='Data points', color='blue')

        val_m, val_u = self.format_with_uncertainty(slope, slope_err if slope_err is not None else 0)
        int_m, int_u = self.format_with_uncertainty(intercept, intercept_err if intercept_err is not None else 0)
        label_text = f"Fit: y=({val_m}±{val_u}) ln(x)+({int_m}±{int_u}), R²={r_squared:.3f}"
        plt.plot(x_fit, y_fit, label=label_text, color='green')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)

        import io
        import base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        return img_str

    def power_fit(self, x_data: List[float], y_data: List[float],
                  y_err: Optional[List[float]] = None) -> Tuple[float, float, float, float, float, float]:
        """
        双对数/幂律拟合：y = C * x^k。在线性化后：ln(y) = k * ln(x) + ln(C)。
        提供 y_err 时，使用对数域的不确定度传播：sigma_Y ≈ y_err / y。
        :param x_data: x 数据（需 > 0）
        :param y_data: y 数据（需 > 0）
        :param y_err: y 的标准不确定度（线性域）
        :return: (k, C, 加权R², k_err, C_err, 减少卡方)
        """
        x = np.array(x_data)
        y = np.array(y_data)

        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("Power-law fit requires all x and y values to be > 0")

        X = np.log(x)
        Y = np.log(y)

        use_weights = y_err is not None and len(y_err) == len(y) and np.all(np.array(y_err) > 0)

        if use_weights:
            sigma_Y = np.array(y_err) / y
            if np.any(sigma_Y <= 0) or np.any(~np.isfinite(sigma_Y)):
                use_weights = False

        if use_weights:
            w = 1.0 / (sigma_Y ** 2)
            S = w.sum()
            Sx = (w * X).sum()
            Sy = (w * Y).sum()
            Sxx = (w * X * X).sum()
            Sxy = (w * X * Y).sum()
            Delta = S * Sxx - Sx * Sx
            k = (S * Sxy - Sx * Sy) / Delta
            lnC = (Sy - k * Sx) / S

            Y_hat = k * X + lnC
            Y_bar_w = Sy / S
            SSE = (w * (Y - Y_hat) ** 2).sum()
            SST = (w * (Y - Y_bar_w) ** 2).sum()
            r_squared_w = 1.0 - (SSE / SST if SST > 0 else 0.0)

            dof = max(1, len(X) - 2)
            chi2_red = SSE / dof
            var_k = S / Delta
            var_lnC = Sxx / Delta
            k_err = float(np.sqrt(var_k * chi2_red))
            lnC_err = float(np.sqrt(var_lnC * chi2_red))
            C = float(np.exp(lnC))
            C_err = float(C * lnC_err)
            return float(k), C, float(r_squared_w), k_err, C_err, float(chi2_red)
        else:
            (k, lnC), cov = np.polyfit(X, Y, 1, cov=True)
            k_err = float(np.sqrt(cov[0, 0])) if cov.size else 0.0
            lnC_err = float(np.sqrt(cov[1, 1])) if cov.size else 0.0
            correlation_matrix = np.corrcoef(X, Y)
            r_squared = float(correlation_matrix[0, 1] ** 2)
            Y_hat = k * X + lnC
            resid = Y - Y_hat
            dof = max(1, len(X) - 2)
            chi2_red = float((resid ** 2).sum() / dof)
            C = float(np.exp(lnC))
            C_err = float(C * lnC_err)
            return float(k), C, float(r_squared), k_err, C_err, chi2_red

    def plot_power_fit(self, x_data: List[float], y_data: List[float],
                       title: str = "Power-Law Fit", save_path: Optional[str] = None,
                       xlabel: str = "X", ylabel: str = "Y",
                       x_err: Optional[List[float]] = None,
                       y_err: Optional[List[float]] = None,
                       k: Optional[float] = None,
                       C: Optional[float] = None,
                       r_squared: Optional[float] = None,
                       k_err: Optional[float] = None,
                       C_err: Optional[float] = None) -> str:
        """
        绘制双对数幂律拟合：y = C x^k，在 log-log 坐标下显示。
        若未提供参数，则内部调用 power_fit 估计，并支持 y_err 加权。
        """
        if k is None or C is None or r_squared is None:
            k, C, r_squared, k_err, C_err, _ = self.power_fit(x_data, y_data, y_err=y_err)
        else:
            k_err = k_err if k_err is not None else 0.0
            C_err = C_err if C_err is not None else 0.0

        x = np.array(x_data)
        x_fit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 100)
        y_fit = C * (x_fit ** k)

        plt.figure(figsize=(10, 6))
        if x_err is not None or y_err is not None:
            plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', label='Data points', color='blue', ecolor='gray', capsize=3)
        else:
            plt.scatter(x_data, y_data, label='Data points', color='blue')

        k_m, k_u = self.format_with_uncertainty(k, k_err if k_err is not None else 0)
        C_m, C_u = self.format_with_uncertainty(C, C_err if C_err is not None else 0)
        label_text = f"Fit: y=({C_m}±{C_u}) x^({k_m}±{k_u}), R²={r_squared:.3f}"
        plt.plot(x_fit, y_fit, label=label_text, color='purple')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, which='both', ls='--')

        if save_path:
            plt.savefig(save_path)

        import io
        import base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        return img_str
