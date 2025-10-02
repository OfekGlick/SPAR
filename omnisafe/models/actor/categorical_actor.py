# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module contains the base categorical distribution actor for the models."""

from abc import ABC, abstractmethod

from omnisafe.models.base import Actor


class CategoricalActor(Actor, ABC):
    """An abstract class for categorical distribution actor.

    A CategoricalActor inherits from Actor and uses Categorical distribution to approximate 
    the policy function for discrete action spaces.

    .. note::
        You can use this class to implement your own actor by inheriting it.
    """

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Get the temperature parameter for softmax (if applicable)."""

    @temperature.setter
    @abstractmethod
    def temperature(self, temp: float) -> None:
        """Set the temperature parameter for softmax (if applicable)."""
